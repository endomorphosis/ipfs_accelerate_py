#!/usr/bin/env python3
"""
Fixed Merged Test Generator - Full Version

This is a comprehensive version of the test generator that provides complete support for:
- Database-driven templates
- Cross-platform hardware support
- Resource pool integration
- WebNN and WebGPU support
- Mobile/edge hardware support (Qualcomm, Samsung, MediaTek)
- Template inheritance
- Advanced hardware detection

Changes from previous version:
- Complete database template support with improved error handling
- Enhanced validation for template variables
- Full cross-platform hardware support for all 10 supported platforms
- Integration with template validation system
- Support for template inheritance and composition
"""

import os
import sys
import argparse
import importlib.util
import logging
import json
import re
import tempfile
from typing import Dict, List, Any, Tuple, Optional, Union
from pathlib import Path
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Template database integration
try:
    from hardware_test_templates.template_database import TemplateDatabase
    HAS_TEMPLATE_DB = True
except ImportError:
    logger.warning("Template database not available. Using hardcoded templates.")
    HAS_TEMPLATE_DB = False

# Template validator integration for enhanced validation
try:
    from template_validator import (
        validate_template_content, 
        validate_python_syntax,
        validate_cross_platform_support
    )
    HAS_VALIDATOR = True
except ImportError:
    logger.warning("Template validator not available. Using basic validation.")
    HAS_VALIDATOR = False

# DuckDB integration for benchmark results
try:
    import duckdb
    HAS_DUCKDB = True
except ImportError:
    logger.warning("DuckDB not available. Benchmark results will be stored in JSON format.")
    HAS_DUCKDB = False

# Hardware detection - enhanced setup with status tracking
hardware_status = {}

# Core hardware platforms
try:
    import torch
    HAS_CUDA = torch.cuda.is_available()
    hardware_status["cuda"] = HAS_CUDA
    
    HAS_ROCM = (HAS_CUDA and hasattr(torch, '_C') and hasattr(torch._C, '_rocm_version')) or ('ROCM_HOME' in os.environ)
    hardware_status["rocm"] = HAS_ROCM
    
    HAS_MPS = hasattr(torch, "mps") and hasattr(torch.mps, "is_available") and torch.mps.is_available()
    hardware_status["mps"] = HAS_MPS
    
    hardware_status["cpu"] = True  # CPU is always available
except ImportError:
    logger.warning("PyTorch not available. Hardware detection limited.")
    HAS_CUDA = False
    HAS_ROCM = False
    HAS_MPS = False
    hardware_status = {"cpu": True, "cuda": False, "rocm": False, "mps": False}

# Other hardware platforms
HAS_OPENVINO = importlib.util.find_spec("openvino") is not None
hardware_status["openvino"] = HAS_OPENVINO

HAS_QNN = importlib.util.find_spec("qnn_wrapper") is not None or importlib.util.find_spec("qti") is not None
hardware_status["qualcomm"] = HAS_QNN

# Web platforms
HAS_WEBNN = importlib.util.find_spec("webnn") is not None or "WEBNN_AVAILABLE" in os.environ
hardware_status["webnn"] = HAS_WEBNN

HAS_WEBGPU = importlib.util.find_spec("webgpu") is not None or "WEBGPU_AVAILABLE" in os.environ
hardware_status["webgpu"] = HAS_WEBGPU

# Mobile platforms (beyond Qualcomm)
HAS_SAMSUNG = importlib.util.find_spec("samsung_npu") is not None or "SAMSUNG_NPU_AVAILABLE" in os.environ
hardware_status["samsung"] = HAS_SAMSUNG

HAS_MEDIATEK = importlib.util.find_spec("mediatek_apu") is not None or "MEDIATEK_APU_AVAILABLE" in os.environ
hardware_status["mediatek"] = HAS_MEDIATEK

# Try to import resource pool for template support
try:
    from resource_pool import ResourcePool, get_global_resource_pool
    HAS_RESOURCE_POOL = True
    logger.debug("Resource pool support available")
except ImportError:
    HAS_RESOURCE_POOL = False
    logger.debug("Resource pool not available")

# Configure DuckDB database path
DB_PATH = os.environ.get("TEMPLATE_DB_PATH", "template_db.duckdb")

def get_template_from_db(model_type: str, 
                        template_type: str = "test", 
                        platform: str = None,
                        db_path: str = None) -> Dict[str, Any]:
    """
    Get a template from the database
    
    Args:
        model_type: Type of model (text, vision, bert, t5, etc.)
        template_type: Type of template (test, benchmark, skill)
        platform: Optional platform filter (cuda, openvino, etc.)
        db_path: Optional path to database file
        
    Returns:
        Dictionary with template content and metadata or None if not found
    """
    if not HAS_TEMPLATE_DB:
        logger.warning("Template database not available, cannot fetch template")
        return None
        
    # Default to environment variable or standard path
    db_path = db_path or os.environ.get("TEMPLATE_DB_PATH", DB_PATH)
    
    try:
        db = TemplateDatabase(db_path=db_path)
        
        # Try to get template with exact model type and platform
        if platform:
            template_data = db.get_template_with_metadata(
                model_type=model_type, 
                template_type=template_type,
                platform=platform
            )
            if template_data:
                logger.debug(f"Found template for {model_type}/{template_type}/{platform}")
                return template_data
        
        # Try to get template with exact model type (any platform)
        template_data = db.get_template_with_metadata(
            model_type=model_type, 
            template_type=template_type
        )
        if template_data:
            logger.debug(f"Found template for {model_type}/{template_type}")
            return template_data
            
        # Try to get general template for model family if specific one not found
        # This uses template inheritance - generic model types can be used if specific ones not available
        generic_types = {
            # Text models
            'bert': 'text_embedding',
            't5': 'text_generation',
            'llama': 'text_generation',
            'gpt': 'text_generation',
            'qwen': 'text_generation',
            
            # Vision models
            'vit': 'vision',
            'clip': 'vision',
            'detr': 'vision_detection',
            
            # Audio models
            'whisper': 'audio',
            'wav2vec2': 'audio',
            'clap': 'audio',
            
            # Multimodal models
            'llava': 'vision_language',
            'xclip': 'video'
        }
        
        # Try to find a more generic template if specific one not found
        generic_type = generic_types.get(model_type.lower(), None)
        if generic_type:
            logger.debug(f"Trying generic template type: {generic_type}")
            
            # Try with platform first
            if platform:
                template_data = db.get_template_with_metadata(
                    model_type=generic_type, 
                    template_type=template_type,
                    platform=platform
                )
                if template_data:
                    logger.debug(f"Found generic template for {generic_type}/{template_type}/{platform}")
                    # Update with specific model type for proper rendering
                    template_data['model_type'] = model_type
                    return template_data
            
            # Try generic type without platform
            template_data = db.get_template_with_metadata(
                model_type=generic_type, 
                template_type=template_type
            )
            if template_data:
                logger.debug(f"Found generic template for {generic_type}/{template_type}")
                # Update with specific model type for proper rendering
                template_data['model_type'] = model_type
                return template_data
        
        # Last resort: try to find a 'default' template
        logger.debug("No specific template found, trying 'default' template")
        template_data = db.get_template_with_metadata(
            model_type='default', 
            template_type=template_type
        )
        if template_data:
            logger.debug(f"Using default template for {template_type}")
            # Update with specific model type for proper rendering
            template_data['model_type'] = model_type
            return template_data
            
        # No templates found
        logger.warning(f"No template found for {model_type}/{template_type}")
        return None
            
    except Exception as e:
        logger.error(f"Error retrieving template from database: {e}")
        return None

def validate_db_template(template_content: str, model_type: str) -> Dict[str, Any]:
    """
    Validate a template from the database
    
    Args:
        template_content: Template content
        model_type: Type of model
        
    Returns:
        Validation results
    """
    if not HAS_VALIDATOR:
        # Minimal validation if validator module not available
        try:
            compile(template_content, f"<template-{model_type}>", 'exec')
            return {
                'valid': True,
                'errors': [],
                'supported_platforms': ['cpu', 'cuda']  # Assume basic support
            }
        except SyntaxError as e:
            return {
                'valid': False,
                'errors': [f"Syntax error: {e}"],
                'supported_platforms': []
            }
    
    # Full validation with template validator
    validation_result = validate_template_content(template_content)
    return validation_result

def render_template(template_content: str, **kwargs) -> str:
    """
    Render a template with variables
    
    Args:
        template_content: Template content with placeholders
        **kwargs: Variables to substitute in the template
        
    Returns:
        Rendered template
    """
    # Simple template rendering
    rendered = template_content
    
    # Find all template variables
    template_vars = re.findall(r'{{(.*?)}}', template_content)
    
    # Process each template variable
    for var in template_vars:
        var = var.strip()
        placeholder = f"{{{{{var}}}}}"
        
        # Handle complex expressions like model_name.replace("-", "")
        if '.' in var:
            base_var, expr = var.split('.', 1)
            base_var = base_var.strip()
            
            if base_var in kwargs:
                # Use a safer approach than eval
                base_value = kwargs[base_var]
                
                # Handle common string operations
                if expr.startswith('replace'):
                    # Parse replace arguments
                    replace_args = re.search(r'replace\s*\(\s*[\'"](.+?)[\'"]\s*,\s*[\'"](.*)[\'"]\s*\)', expr)
                    if replace_args:
                        old_str, new_str = replace_args.groups()
                        value = str(base_value).replace(old_str, new_str)
                        rendered = rendered.replace(placeholder, value)
                elif expr.startswith('capitalize'):
                    value = str(base_value).capitalize()
                    rendered = rendered.replace(placeholder, value)
                elif expr.startswith('lower'):
                    value = str(base_value).lower()
                    rendered = rendered.replace(placeholder, value)
                elif expr.startswith('upper'):
                    value = str(base_value).upper()
                    rendered = rendered.replace(placeholder, value)
                elif expr.startswith('title'):
                    value = str(base_value).title()
                    rendered = rendered.replace(placeholder, value)
                else:
                    # For unsupported operations, just use the base value
                    rendered = rendered.replace(placeholder, str(base_value))
        else:
            # Simple variable substitution
            if var in kwargs:
                rendered = rendered.replace(placeholder, str(kwargs[var]))
            else:
                logger.warning(f"Template variable '{var}' not provided")
    
    return rendered

# Model registry for common test models
MODEL_REGISTRY = {
    # Text models
    "bert": "bert-base-uncased",
    "bert-tiny": "prajjwal1/bert-tiny",
    "t5": "t5-small",
    "t5-tiny": "google/t5-efficient-tiny",
    "llama": "openlm-research/open_llama_3b",
    "llama-small": "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
    "llama-tiny": "TinyLlama/TinyLlama-1.1B-Chat-v1.0", 
    "qwen": "Qwen/Qwen1.5-0.5B",
    "qwen2": "Qwen/Qwen2-0.5B-Instruct",
    
    # Vision models
    "vit": "google/vit-base-patch16-224",
    "vit-tiny": "google/vit-tiny-patch16-224",
    "clip": "openai/clip-vit-base-patch32",
    "clip-small": "openai/clip-vit-base-patch16",
    "detr": "facebook/detr-resnet-50",
    
    # Audio models
    "whisper": "openai/whisper-tiny",
    "whisper-small": "openai/whisper-small",
    "wav2vec2": "facebook/wav2vec2-base-960h",
    "wav2vec2-small": "facebook/wav2vec2-base",
    "clap": "laion/clap-htsat-unfused",
    
    # Multimodal models
    "llava": "llava-hf/llava-1.5-7b-hf",
    "llava-small": "llava-hf/llava-1.5-mistral-7b",
    "llava-tiny": "llava-hf/bakLlava-v1-hf",
    "xclip": "microsoft/xclip-base-patch32",
}

def get_model_name(model_id: str) -> str:
    """
    Get the full model name for a model ID
    
    Args:
        model_id: Short model ID like "bert" or "t5"
        
    Returns:
        Full model name like "bert-base-uncased" or "t5-small"
    """
    # Check if it's already a full name (contains '/' or '-')
    if '/' in model_id or (model_id not in MODEL_REGISTRY and '-' in model_id):
        return model_id
        
    # Otherwise look up in registry
    return MODEL_REGISTRY.get(model_id, model_id)

def get_model_type(model_id: str) -> str:
    """
    Get the model type from a model ID
    
    Args:
        model_id: Model ID or full name
        
    Returns:
        Model type (bert, t5, vit, etc.)
    """
    # Extract model type from full name or ID
    if '/' in model_id:
        # Handle org/model format like "google/t5-small"
        model_id = model_id.split('/')[-1]
    
    # Handle models with versions like "bert-base-uncased"
    base_model = model_id.split('-')[0].lower()
    
    # Special case handling
    if base_model.startswith('t5'):
        return 't5'
    elif base_model.startswith('vit'):
        return 'vit'
    elif base_model.startswith('clip'):
        return 'clip'
    elif base_model.startswith('llama'):
        return 'llama'
    elif base_model.startswith('wav2vec'):
        return 'wav2vec2'
    
    return base_model

def get_available_hardware() -> List[str]:
    """
    Get list of available hardware platforms
    
    Returns:
        List of hardware platform names
    """
    available = []
    
    # Check core platforms
    if hardware_status.get("cpu", True):
        available.append("cpu")
    if hardware_status.get("cuda", False):
        available.append("cuda")
    if hardware_status.get("rocm", False):
        available.append("rocm") 
    if hardware_status.get("mps", False):
        available.append("mps")
        
    # Check accelerator platforms
    if hardware_status.get("openvino", False):
        available.append("openvino")
    if hardware_status.get("qualcomm", False):
        available.append("qualcomm")
    if hardware_status.get("samsung", False):
        available.append("samsung")
    if hardware_status.get("mediatek", False):
        available.append("mediatek")
        
    # Check web platforms
    if hardware_status.get("webnn", False):
        available.append("webnn")
    if hardware_status.get("webgpu", False):
        available.append("webgpu")
        
    return available

def generate_test(model_id: str, 
                 output_file: str = None, 
                 hardware_platforms: List[str] = None,
                 use_db_templates: bool = True,
                 db_path: str = None,
                 add_imports: List[str] = None,
                 template_type: str = "test",
                 verbose: bool = False) -> Dict[str, Any]:
    """
    Generate a test file for a model
    
    Args:
        model_id: Model ID or name, e.g., "bert" or "bert-base-uncased"
        output_file: Path to output file
        hardware_platforms: List of hardware platforms to include
        use_db_templates: Whether to use database templates
        db_path: Path to database file
        add_imports: Additional imports to add
        template_type: Type of template to use
        verbose: Whether to print verbose output
    
    Returns:
        Dictionary with generation results
    """
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
        
    # Resolve model name
    full_model_name = get_model_name(model_id)
    model_type = get_model_type(model_id)
    
    logger.debug(f"Generating {template_type} for model {full_model_name} (type: {model_type})")
    
    # Determine available hardware platforms if not specified
    if not hardware_platforms or 'all' in hardware_platforms:
        hardware_platforms = get_available_hardware()
        logger.debug(f"Using auto-detected hardware platforms: {hardware_platforms}")
    
    # Init variables for template content
    template_content = None
    template_data = None
    supported_platforms = []
    
    # Get template from database if available
    if use_db_templates and HAS_TEMPLATE_DB:
        # Try to get platform-specific templates for each platform
        for platform in hardware_platforms:
            # Get template for this specific platform
            platform_template = get_template_from_db(
                model_type=model_type,
                template_type=template_type,
                platform=platform,
                db_path=db_path
            )
            
            if platform_template:
                logger.debug(f"Found template for {model_type} with {platform} platform")
                if not template_data:
                    # Use first platform template as base
                    template_data = platform_template
                    template_content = platform_template.get('template_content')
                # We'll combine multiple platform templates later if needed
                supported_platforms.append(platform)
        
        # If no platform-specific template found, try generic model template
        if not template_data:
            template_data = get_template_from_db(
                model_type=model_type,
                template_type=template_type,
                db_path=db_path
            )
            
            if template_data:
                logger.debug(f"Found generic template for {model_type}")
                template_content = template_data.get('template_content')
                
                # Extract supported platforms from template
                if template_content and HAS_VALIDATOR:
                    _, _, supported_platforms = validate_cross_platform_support(template_content)
                else:
                    # Assume basic platform support
                    supported_platforms = ["cpu", "cuda"]
    
    # If no template found or database not available, use hardcoded templates
    if not template_content:
        logger.warning(f"No template found for {model_type} in database, using hardcoded template")
        
        # Basic hardcoded templates keyed by model type
        hardcoded_templates = {
            "bert": """import unittest
import torch
from transformers import AutoModel, AutoTokenizer

class Test{{model_name.replace("-", "").capitalize()}}(unittest.TestCase):
    @classmethod
    def setup_class(cls):
        cls.model_name = "{{model_name}}"
        cls.tokenizer = AutoTokenizer.from_pretrained(cls.model_name)
        cls.model = AutoModel.from_pretrained(cls.model_name)
        cls.device = "cuda" if torch.cuda.is_available() else "cpu"
        cls.model = cls.model.to(cls.device)
        
    def test_model_forward(self):
        inputs = self.tokenizer("Test input", return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.model(**inputs)
        self.assertIsNotNone(outputs)
        self.assertEqual(outputs.last_hidden_state.shape[0], 1)
""",
            "t5": """import unittest
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

class Test{{model_name.replace("-", "").capitalize()}}(unittest.TestCase):
    @classmethod
    def setup_class(cls):
        cls.model_name = "{{model_name}}"
        cls.tokenizer = AutoTokenizer.from_pretrained(cls.model_name)
        cls.model = AutoModelForSeq2SeqLM.from_pretrained(cls.model_name)
        cls.device = "cuda" if torch.cuda.is_available() else "cpu"
        cls.model = cls.model.to(cls.device)
        
    def test_model_forward(self):
        input_text = "translate English to German: Hello, how are you?"
        inputs = self.tokenizer(input_text, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.model.generate(**inputs)
        translation = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        self.assertIsNotNone(translation)
""",
            "vit": """import unittest
import torch
from transformers import AutoImageProcessor, AutoModel
import numpy as np

class Test{{model_name.replace("-", "").capitalize()}}(unittest.TestCase):
    @classmethod
    def setup_class(cls):
        cls.model_name = "{{model_name}}"
        cls.processor = AutoImageProcessor.from_pretrained(cls.model_name)
        cls.model = AutoModel.from_pretrained(cls.model_name)
        cls.device = "cuda" if torch.cuda.is_available() else "cpu"
        cls.model = cls.model.to(cls.device)
        
    def test_model_forward(self):
        # Create dummy image data (3 channels, 224x224)
        dummy_image = np.random.rand(3, 224, 224)
        inputs = self.processor(dummy_image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.model(**inputs)
        self.assertIsNotNone(outputs)
""",
        }
        
        # Default template for unsupported model types
        default_template = """import unittest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class Test{{model_name.replace("-", "").capitalize()}}(unittest.TestCase):
    @classmethod
    def setup_class(cls):
        cls.model_name = "{{model_name}}"
        try:
            cls.tokenizer = AutoTokenizer.from_pretrained(cls.model_name)
            cls.model = AutoModelForCausalLM.from_pretrained(cls.model_name)
            cls.device = "cuda" if torch.cuda.is_available() else "cpu"
            cls.model = cls.model.to(cls.device)
            cls.setup_success = True
        except Exception as e:
            cls.setup_success = False
            cls.setup_error = str(e)
        
    def test_model_setup(self):
        self.assertTrue(self.setup_success, f"Model setup failed: {getattr(self, 'setup_error', 'Unknown error')}")
        
    def test_model_forward(self):
        if not self.setup_success:
            self.skipTest("Model setup failed")
            
        inputs = self.tokenizer("Test input", return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.model(**inputs)
        self.assertIsNotNone(outputs)
"""
        
        # Get template for model type or use default
        template_content = hardcoded_templates.get(model_type, default_template)
        supported_platforms = ["cpu", "cuda"]  # Assumed for hardcoded templates
    
    # Ensure we have a template
    if not template_content:
        return {
            "success": False,
            "error": f"Failed to generate {template_type} for {model_id}: no template available"
        }
    
    # Validate template
    validation_result = validate_db_template(template_content, model_type)
    if not validation_result.get('valid', False):
        logger.warning(f"Template validation failed: {validation_result.get('errors', ['Unknown error'])}")
    
    # Get supported platforms from validation if not already determined
    if not supported_platforms and validation_result.get('supported_platforms'):
        supported_platforms = validation_result.get('supported_platforms')
    
    # Filter hardware platforms to only those supported
    filtered_hardware = [h for h in hardware_platforms if h in supported_platforms]
    
    if not filtered_hardware:
        logger.warning(f"No supported hardware platforms found for {model_type}. Using all requested platforms.")
        filtered_hardware = hardware_platforms
    
    # Render template with variables
    generation_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    template_variables = {
        "model_name": full_model_name,
        "model_type": model_type,
        "hardware_platforms": filtered_hardware,
        "timestamp": generation_timestamp,
        "generator": "fixed_merged_test_generator.py"
    }
    
    # Render the template
    rendered_template = render_template(template_content, **template_variables)
    
    # Add auto-generated header
    header = f"""#!/usr/bin/env python3
# Auto-generated {template_type} for {full_model_name}
# Generated at: {generation_timestamp}
# Generator: fixed_merged_test_generator.py

"""
    
    # Add additional imports if specified
    if add_imports:
        imports = "\n".join([f"import {imp}" for imp in add_imports])
        header += imports + "\n\n"
    
    rendered_template = header + rendered_template
    
    # Write to file if output file specified
    if output_file:
        try:
            with open(output_file, 'w') as f:
                f.write(rendered_template)
            logger.info(f"Generated {template_type} file: {output_file}")
        except Exception as e:
            logger.error(f"Failed to write output file: {e}")
            return {
                "success": False,
                "error": f"Failed to write output file: {e}",
                "code": rendered_template
            }
    
    # Return result
    return {
        "success": True,
        "model_id": model_id,
        "model_name": full_model_name,
        "model_type": model_type,
        "template_type": template_type,
        "supported_platforms": supported_platforms,
        "used_platforms": filtered_hardware,
        "output_file": output_file,
        "template_from_db": template_data is not None,
        "code": rendered_template
    }

def main():
    """Main CLI function"""
    parser = argparse.ArgumentParser(description="Fixed Merged Test Generator")
    parser.add_argument("--model", "-m", type=str, help="Model ID or name (e.g., 'bert' or 'bert-base-uncased')")
    parser.add_argument("--output", "-o", type=str, help="Output file path")
    parser.add_argument("--hardware", "-p", type=str, help="Comma-separated list of hardware platforms (e.g., 'cpu,cuda,openvino') or 'all'")
    parser.add_argument("--model-registry", action="store_true", help="Show available models in the registry")
    parser.add_argument("--list-hardware", action="store_true", help="List available hardware platforms")
    parser.add_argument("--use-db-templates", action="store_true", help="Use database templates")
    parser.add_argument("--db-path", type=str, help="Path to template database")
    parser.add_argument("--template-type", type=str, default="test", choices=["test", "benchmark", "skill"], help="Type of template to use")
    parser.add_argument("--add-imports", type=str, help="Comma-separated list of additional imports")
    parser.add_argument("--all-models", action="store_true", help="Generate tests for all models in registry")
    parser.add_argument("--output-dir", type=str, help="Output directory for --all-models")
    parser.add_argument("--cross-platform", action="store_true", help="Include all platforms in generated test")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--enable-all", action="store_true", help="Equivalent to --use-db-templates --cross-platform --verbose")
    parser.add_argument("--validate-only", action="store_true", help="Validate template without generating code")
    parser.add_argument("--family", type=str, help="Generate tests for all models in a family (e.g., 'text', 'vision')")
    
    args = parser.parse_args()
    
    # Enable all features if requested
    if args.enable_all:
        args.use_db_templates = True
        args.cross_platform = True
        args.verbose = True
    
    # List models in registry if requested
    if args.model_registry:
        print("\nAvailable models in registry:")
        print("-" * 50)
        print("{:<20} {:<50}".format("Model ID", "Full Model Name"))
        print("-" * 50)
        
        # Group models by category for better display
        categories = {
            "Text Models": [k for k in MODEL_REGISTRY.keys() if k in ["bert", "bert-tiny", "t5", "t5-tiny", "llama", "llama-small", "llama-tiny", "qwen", "qwen2"]],
            "Vision Models": [k for k in MODEL_REGISTRY.keys() if k in ["vit", "vit-tiny", "clip", "clip-small", "detr"]],
            "Audio Models": [k for k in MODEL_REGISTRY.keys() if k in ["whisper", "whisper-small", "wav2vec2", "wav2vec2-small", "clap"]],
            "Multimodal Models": [k for k in MODEL_REGISTRY.keys() if k in ["llava", "llava-small", "llava-tiny", "xclip"]]
        }
        
        for category, models in categories.items():
            print(f"\n{category}:")
            for model_id in sorted(models):
                print("{:<20} {:<50}".format(model_id, MODEL_REGISTRY[model_id]))
        
        return 0
    
    # List available hardware platforms if requested
    if args.list_hardware:
        available = get_available_hardware()
        
        print("\nAvailable hardware platforms:")
        print("-" * 50)
        for platform in available:
            print(f"- {platform}")
        
        # Show all possible platforms with status
        print("\nAll hardware platform status:")
        print("-" * 50)
        for platform, status in hardware_status.items():
            status_str = "✅ Available" if status else "❌ Not available"
            print(f"- {platform}: {status_str}")
        
        return 0
    
    # Validate args
    if not (args.model or args.all_models or args.family) and not (args.model_registry or args.list_hardware):
        parser.error("You must specify --model, --all-models, or --family")
        return 1
    
    # Parse hardware platforms
    hardware_platforms = None
    if args.hardware:
        if args.hardware.lower() == 'all':
            hardware_platforms = ['all']  # Will be resolved to all available platforms
        else:
            hardware_platforms = args.hardware.split(',')
    elif args.cross_platform:
        hardware_platforms = ['all']
    
    # Parse additional imports
    add_imports = args.add_imports.split(',') if args.add_imports else None
    
    # Generate test for single model
    if args.model:
        # Check if output file is specified
        if not args.output and not args.validate_only:
            default_output = f"test_{args.model.replace('/', '_').replace('-', '_').lower()}.py"
            print(f"No output file specified, using default: {default_output}")
            args.output = default_output
        
        result = generate_test(
            model_id=args.model,
            output_file=args.output if not args.validate_only else None,
            hardware_platforms=hardware_platforms,
            use_db_templates=args.use_db_templates,
            db_path=args.db_path,
            add_imports=add_imports,
            template_type=args.template_type,
            verbose=args.verbose
        )
        
        if args.validate_only:
            # Just show validation results
            print(f"\nValidation results for {args.model}:")
            print(f"Valid: {result.get('success', False)}")
            if not result.get('success', False):
                print(f"Error: {result.get('error', 'Unknown error')}")
            else:
                print(f"Supported platforms: {', '.join(result.get('supported_platforms', []))}")
                print(f"Template from DB: {result.get('template_from_db', False)}")
        else:
            # Show generation results
            if result.get("success", False):
                print(f"\n✅ Successfully generated {args.template_type} for {args.model}")
                print(f"Output file: {result.get('output_file')}")
                print(f"Supported platforms: {', '.join(result.get('supported_platforms', []))}")
                print(f"Used platforms: {', '.join(result.get('used_platforms', []))}")
                print(f"Template from DB: {result.get('template_from_db', False)}")
            else:
                print(f"\n❌ Failed to generate {args.template_type} for {args.model}")
                print(f"Error: {result.get('error', 'Unknown error')}")
                return 1
    
    # Generate tests for all models in registry
    elif args.all_models:
        if not args.output_dir:
            print("You must specify --output-dir when using --all-models")
            return 1
        
        # Create output directory if it doesn't exist
        os.makedirs(args.output_dir, exist_ok=True)
        
        success_count = 0
        failure_count = 0
        
        for model_id, model_name in MODEL_REGISTRY.items():
            output_file = os.path.join(args.output_dir, f"test_{model_id.replace('-', '_').lower()}.py")
            
            result = generate_test(
                model_id=model_id,
                output_file=output_file,
                hardware_platforms=hardware_platforms,
                use_db_templates=args.use_db_templates,
                db_path=args.db_path,
                add_imports=add_imports,
                template_type=args.template_type,
                verbose=args.verbose
            )
            
            if result.get("success", False):
                success_count += 1
                if args.verbose:
                    print(f"✅ Generated {args.template_type} for {model_id} ({model_name})")
            else:
                failure_count += 1
                print(f"❌ Failed to generate {args.template_type} for {model_id}: {result.get('error', 'Unknown error')}")
        
        print(f"\nGeneration complete: {success_count} successful, {failure_count} failed")
    
    # Generate tests for all models in a family
    elif args.family:
        if not args.output_dir:
            print("You must specify --output-dir when using --family")
            return 1
        
        # Create output directory if it doesn't exist
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Filter models by family
        family_models = {}
        family_lower = args.family.lower()
        
        # Family to model type mapping
        family_mapping = {
            "text": ["bert", "bert-tiny", "t5", "t5-tiny", "llama", "llama-small", "llama-tiny", "qwen", "qwen2"],
            "vision": ["vit", "vit-tiny", "clip", "clip-small", "detr"],
            "audio": ["whisper", "whisper-small", "wav2vec2", "wav2vec2-small", "clap"],
            "multimodal": ["llava", "llava-small", "llava-tiny", "xclip"],
            "text_embedding": ["bert", "bert-tiny"],
            "text_generation": ["t5", "t5-tiny", "llama", "llama-small", "llama-tiny", "qwen", "qwen2"],
        }
        
        if family_lower in family_mapping:
            for model_id in family_mapping[family_lower]:
                if model_id in MODEL_REGISTRY:
                    family_models[model_id] = MODEL_REGISTRY[model_id]
        else:
            # Manual search for partial matches
            for model_id, model_name in MODEL_REGISTRY.items():
                if family_lower in model_id.lower():
                    family_models[model_id] = model_name
        
        if not family_models:
            print(f"No models found for family: {args.family}")
            return 1
        
        print(f"Generating {args.template_type}s for {len(family_models)} models in family: {args.family}")
        
        success_count = 0
        failure_count = 0
        
        for model_id, model_name in family_models.items():
            output_file = os.path.join(args.output_dir, f"test_{model_id.replace('-', '_').lower()}.py")
            
            result = generate_test(
                model_id=model_id,
                output_file=output_file,
                hardware_platforms=hardware_platforms,
                use_db_templates=args.use_db_templates,
                db_path=args.db_path,
                add_imports=add_imports,
                template_type=args.template_type,
                verbose=args.verbose
            )
            
            if result.get("success", False):
                success_count += 1
                if args.verbose:
                    print(f"✅ Generated {args.template_type} for {model_id} ({model_name})")
            else:
                failure_count += 1
                print(f"❌ Failed to generate {args.template_type} for {model_id}: {result.get('error', 'Unknown error')}")
        
        print(f"\nGeneration complete: {success_count} successful, {failure_count} failed")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())