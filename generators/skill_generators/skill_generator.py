#!/usr/bin/env python3
"""
Integrated Skillset Generator - Enhanced Version

This version of the skillset generator includes full support for:
- Database-driven templates
- Cross-platform hardware support
- Template inheritance
- Resource pool integration
- WebNN and WebGPU support
- Mobile/edge hardware support (Qualcomm, Samsung, MediaTek)
- Advanced template variable handling

Compared to previous version:
- Complete database template support with improved error handling
- Enhanced validation for template variables
- Full cross-platform hardware support
- Integration with template validation system
- Support for template inheritance and composition
"""

import os
import sys
import argparse
import importlib.util
import logging
import re
import json
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
    HAS_TEMPLATE_DB = False
    logger.warning("Template database not available. Using hardcoded templates.")

# Template validator integration
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

# Model registry for common test models
MODEL_REGISTRY = {
    # Text models
    "bert": "bert-base-uncased",
    "bert-tiny": "prajjwal1/bert-tiny",
    "t5": "t5-small",
    "t5-tiny": "google/t5-efficient-tiny",
    "llama": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "llama-small": "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
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
    "wav2vec2": "facebook/wav2vec2-base",
    "wav2vec2-small": "facebook/wav2vec2-base-960h",
    "clap": "laion/clap-htsat-unfused",
    
    # Multimodal models
    "llava": "llava-hf/llava-1.5-7b-hf",
    "llava-small": "llava-hf/llava-1.5-mistral-7b",
    "llava-tiny": "llava-hf/bakLlava-v1-hf",
    "xclip": "microsoft/xclip-base-patch32",
    "llava_next": "llava-hf/llava-1.6-34b-hf"
}

# Define key model hardware support
KEY_MODEL_HARDWARE_MAP = {
    "bert": {"cpu": "REAL", "cuda": "REAL", "rocm": "REAL", "mps": "REAL", "openvino": "REAL", "qualcomm": "REAL", "webnn": "REAL", "webgpu": "REAL"},
    "t5": {"cpu": "REAL", "cuda": "REAL", "rocm": "REAL", "mps": "REAL", "openvino": "REAL", "qualcomm": "REAL", "webnn": "REAL", "webgpu": "REAL"},
    "vit": {"cpu": "REAL", "cuda": "REAL", "rocm": "REAL", "mps": "REAL", "openvino": "REAL", "qualcomm": "REAL", "webnn": "REAL", "webgpu": "REAL"},
    "clip": {"cpu": "REAL", "cuda": "REAL", "rocm": "REAL", "mps": "REAL", "openvino": "REAL", "qualcomm": "REAL", "webnn": "REAL", "webgpu": "REAL"},
    "whisper": {"cpu": "REAL", "cuda": "REAL", "rocm": "REAL", "mps": "REAL", "openvino": "REAL", "qualcomm": "REAL", "webnn": "SIMULATION", "webgpu": "SIMULATION"},
    "wav2vec2": {"cpu": "REAL", "cuda": "REAL", "rocm": "REAL", "mps": "REAL", "openvino": "REAL", "qualcomm": "REAL", "webnn": "SIMULATION", "webgpu": "SIMULATION"},
    "clap": {"cpu": "REAL", "cuda": "REAL", "rocm": "REAL", "mps": "REAL", "openvino": "REAL", "qualcomm": "REAL", "webnn": "SIMULATION", "webgpu": "SIMULATION"},
    "llama": {"cpu": "REAL", "cuda": "REAL", "rocm": "REAL", "mps": "REAL", "openvino": "REAL", "qualcomm": "REAL", "webnn": "SIMULATION", "webgpu": "SIMULATION"},
    "llava": {"cpu": "REAL", "cuda": "REAL", "rocm": "SIMULATION", "mps": "SIMULATION", "openvino": "SIMULATION", "qualcomm": "SIMULATION", "webnn": "SIMULATION", "webgpu": "SIMULATION"},
    "xclip": {"cpu": "REAL", "cuda": "REAL", "rocm": "REAL", "mps": "REAL", "openvino": "REAL", "qualcomm": "REAL", "webnn": "SIMULATION", "webgpu": "SIMULATION"},
    "detr": {"cpu": "REAL", "cuda": "REAL", "rocm": "REAL", "mps": "REAL", "openvino": "REAL", "qualcomm": "REAL", "webnn": "SIMULATION", "webgpu": "SIMULATION"},
    "qwen2": {"cpu": "REAL", "cuda": "REAL", "rocm": "SIMULATION", "mps": "SIMULATION", "openvino": "SIMULATION", "qualcomm": "SIMULATION", "webnn": "SIMULATION", "webgpu": "SIMULATION"}
}

def get_template_from_db(model_type: str, 
                        template_type: str = "skill", 
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

def detect_model_modality(model_name: str) -> str:
    """
    Detect the modality of a model from its name.
    
    Args:
        model_name: Name of the model
        
    Returns:
        Modality of the model (text, vision, audio, multimodal, video)
    """
    model_lower = model_name.lower()
    
    # Text models
    if any(t in model_lower for t in ["bert", "gpt", "t5", "llama", "roberta", "qwen"]):
        return "text"
    
    # Vision models
    if any(v in model_lower for v in ["vit", "deit", "resnet", "convnext", "detr"]):
        return "vision"
    
    # Audio models
    if any(a in model_lower for t in ["wav2vec", "whisper", "hubert", "clap"]):
        return "audio"
    
    # Multimodal models
    if any(m in model_lower for t in ["clip", "llava", "blip"]):
        return "multimodal"
    
    # Video models
    if any(v in model_lower for t in ["xclip", "video"]):
        return "video"
    
    # Default to text
    return "text"

def get_model_family(model_name: str) -> str:
    """
    Get the model family from a model name.
    
    Args:
        model_name: Name of the model
        
    Returns:
        Model family
    """
    model_lower = model_name.lower()
    
    # Extract base model name without version
    if '-' in model_lower:
        base_model = model_lower.split('-')[0]
    elif '/' in model_lower:
        # Handle org/model format like "google/t5-small"
        base_model = model_lower.split('/')[-1].split('-')[0]
    else:
        base_model = model_lower
    
    # Specific model families
    if base_model == 'bert' or base_model == 'roberta':
        return 'text_embedding'
    elif base_model == 't5' or base_model.startswith('t5'):
        return 'text_generation'
    elif base_model == 'gpt' or base_model.startswith('gpt'):
        return 'text_generation'
    elif base_model == 'llama' or base_model.startswith('llama'):
        return 'text_generation'
    elif base_model == 'qwen' or base_model.startswith('qwen'):
        return 'text_generation'
    elif base_model == 'vit' or base_model.startswith('vit'):
        return 'vision'
    elif base_model == 'clip' or base_model.startswith('clip'):
        return 'vision'
    elif base_model == 'whisper' or base_model.startswith('whisper'):
        return 'audio'
    elif base_model == 'wav2vec2' or base_model.startswith('wav2vec'):
        return 'audio'
    elif base_model == 'clap' or base_model.startswith('clap'):
        return 'audio'
    elif base_model == 'llava' or base_model.startswith('llava'):
        return 'vision_language'
    elif base_model == 'xclip' or base_model.startswith('xclip'):
        return 'video'
    elif base_model == 'detr' or base_model.startswith('detr'):
        return 'vision_detection'
    
    # Default to modality
    return detect_model_modality(model_name)

def resolve_model_name(model_name: str) -> str:
    """
    Resolve model name to get the full model ID if it's a short name.
    
    Args:
        model_name: Short model name or full model ID
        
    Returns:
        Full model ID
    """
    # If it's a key in MODEL_REGISTRY, return the full model ID
    model_base = model_name.split("-")[0].lower() if "-" in model_name else model_name.lower()
    if model_base in MODEL_REGISTRY:
        return MODEL_REGISTRY[model_base]
    # Otherwise, return the model name as is
    return model_name

def validate_template(template_content: str, model_type: str) -> Dict[str, Any]:
    """
    Validate a template.
    
    Args:
        template_content: Template content
        model_type: Type of model
        
    Returns:
        Validation results
    """
    # Basic validation - no matter what, we'll try this
    try:
        compile(template_content, f"<template-{model_type}>", 'exec')
        basic_result = {
            'valid': True,
            'errors': [],
            'supported_platforms': ['cpu', 'cuda']  # Assume basic support
        }
    except SyntaxError as e:
        basic_result = {
            'valid': False,
            'errors': [f"Syntax error: {e}"],
            'supported_platforms': []
        }
    
    # If validator not available, return basic validation
    if not HAS_VALIDATOR:
        return basic_result
        
    # Try advanced validation, but fall back if it fails
    try:
        validation_result = validate_template_content(template_content)
        return validation_result
    except Exception as e:
        logger.warning(f"Error validating template with validator: {e}")
        # Fallback to basic validation
        return basic_result

def render_template(template_content: str, **kwargs) -> str:
    """
    Render a template with variables.
    
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

def get_available_hardware() -> List[str]:
    """
    Get list of available hardware platforms.
    
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

def generate_skill_file(model_name: str,
                      platform: str = None,
                      output_dir: str = None,
                      use_db_templates: bool = False,
                      db_path: str = None,
                      cross_platform: bool = False,
                      verbose: bool = False) -> Dict[str, Any]:
    """
    Generate a skill file for a model.
    
    Args:
        model_name: Model name or ID
        platform: Comma-separated list of hardware platforms or 'all'
        output_dir: Output directory for generated file
        use_db_templates: Whether to use database templates
        db_path: Path to template database file
        cross_platform: Whether to include all platform support
        verbose: Whether to enable verbose logging
        
    Returns:
        Dictionary with generation results
    """
    try:
        logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    
        # Get model information
        model_type = get_model_family(model_name)
        modality = detect_model_modality(model_name)
        resolved_model_id = resolve_model_name(model_name)
        
        # Extract base model name without version
        model_base = model_name.split("-")[0].lower() if "-" in model_name else model_name.lower()
        
        logger.debug(f"Generating skill for model {resolved_model_id} (type: {model_type}, modality: {modality})")
    
        # Determine hardware platforms to include
        if cross_platform or (platform and platform.lower() == 'all'):
            hardware_platforms = get_available_hardware()
            logger.debug(f"Using all available hardware platforms: {hardware_platforms}")
        elif platform:
            hardware_platforms = [p.strip() for p in platform.split(",")]
            logger.debug(f"Using specified hardware platforms: {hardware_platforms}")
        else:
            # Default to common platforms
            hardware_platforms = ["cpu", "cuda"]
            if hardware_status.get("mps", False):
                hardware_platforms.append("mps")
            logger.debug(f"Using default hardware platforms: {hardware_platforms}")
        
        # Initialize template variables for generation
        template_content = None
        template_data = None
        supported_platforms = []
    
        # Get template from database if enabled
        if use_db_templates and HAS_TEMPLATE_DB:
            # Try to get platform-specific templates
            for p in hardware_platforms:
                platform_template = get_template_from_db(
                    model_type=model_type,
                    template_type="skill",
                    platform=p,
                    db_path=db_path
                )
                
                if platform_template:
                    logger.debug(f"Found template for {model_type} with {p} platform")
                    if not template_data:
                        # Use first platform template as base
                        template_data = platform_template
                        template_content = platform_template.get('template_content')
                    # We'll combine platforms later if needed
                    supported_platforms.append(p)
            
            # If no platform-specific template, try generic model template
            if not template_data:
                template_data = get_template_from_db(
                    model_type=model_type,
                    template_type="skill",
                    db_path=db_path
                )
                
                if template_data:
                    logger.debug(f"Found generic template for {model_type}")
                    template_content = template_data.get('template_content')
                    
                    # Extract supported platforms from template
                    if template_content and HAS_VALIDATOR:
                        try:
                            # This function may not be available or might use self incorrectly
                            _, _, platforms = validate_cross_platform_support(template_content)
                            if platforms:
                                supported_platforms = platforms
                        except Exception as e:
                            logger.warning(f"Error validating cross-platform support: {e}")
                            # Fallback to basic support assumption
                            supported_platforms = ["cpu", "cuda"]
                    else:
                        # Assume basic platform support
                        supported_platforms = ["cpu", "cuda"]
    
        # If no database template found, try legacy method
        if not template_content and use_db_templates and HAS_TEMPLATE_DB:
            logger.debug("Trying legacy method to fetch template")
            try:
                legacy_template_data = get_template_from_db(
                    model_type=model_base,
                    template_type="skill",
                    db_path=db_path
                )
                if legacy_template_data and isinstance(legacy_template_data, dict):
                    template_content = legacy_template_data.get('template_content')
                # If it returned just a string
                elif legacy_template_data and isinstance(legacy_template_data, str):
                    template_content = legacy_template_data
                    
                if template_content:
                    logger.debug(f"Found template using legacy method for {model_base}")
                    supported_platforms = ["cpu", "cuda"]  # Assume basic support for legacy templates
            except Exception as e:
                logger.warning(f"Error fetching legacy template: {e}")
                template_content = None
        
        # Create file name and path
        file_name = f"skill_hf_{model_name.replace('-', '_').replace('/', '_')}.py"
        
        # Use output_dir if specified, otherwise use current directory
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, file_name)
        else:
            output_path = file_name
        
        # Validate template if available
        if template_content and HAS_VALIDATOR:
            validation_result = validate_template(template_content, model_type)
            if not validation_result.get('valid', False):
                logger.warning(f"Template validation failed: {validation_result.get('errors', ['Unknown error'])}")
        
        # Generate file content based on template or fallback to default
        if template_content and use_db_templates:
            try:
                # Prepare context variables for template rendering
                generation_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                # Create context for rendering
                context = {
                    "model_name": model_name,
                    "model_id": resolved_model_id,
                    "model_type": model_type,
                    "model_base": model_base,
                    "platforms": ",".join(hardware_platforms),
                    "supported_platforms": ",".join(supported_platforms),
                    "modality": modality,
                    "class_name": f"{model_name.replace('-', '').replace('/', '').title()}Skill",
                    "timestamp": generation_timestamp,
                    "generator": "integrated_skillset_generator.py"
                }
                
                # Debug info to track what might be causing error
                logger.debug(f"Context for rendering: {str(context)}")
                logger.debug(f"Template content length: {len(template_content) if template_content else 0}")
                
                # Render the template
                try:
                    file_content = render_template(template_content, **context)
                except Exception as render_error:
                    logger.error(f"Error rendering template: {render_error}")
                    logger.error(f"Template content: {template_content[:100]}...")
                    raise render_error
                
                # Add auto-generated header
                header = f"""#!/usr/bin/env python3
# Auto-generated skill file for {resolved_model_id}
# Generated at: {generation_timestamp}
# Generator: integrated_skillset_generator.py

"""
                file_content = header + file_content
                
                # Write template-based content
                with open(output_path, "w") as f:
                    f.write(file_content)
                
                logger.info(f"Generated skill file from template: {output_path}")
                
                return {
                    "success": True,
                    "output_path": output_path,
                    "model_name": model_name,
                    "model_id": resolved_model_id,
                    "template_from_db": True,
                    "supported_platforms": supported_platforms
                }
            except Exception as e:
                logger.error(f"Error applying template: {e}")
                logger.info("Falling back to default generation")
                # Fall back to default generation if template fails
                template_content = None
    
        # Default generation if no template or template failed
        try:
            with open(output_path, "w") as f:
                f.write(f'''#!/usr/bin/env python3
"""
Skill implementation for {model_name} with hardware platform support
"""

import os
import sys
import torch
import numpy as np
import logging
from transformers import AutoModel, AutoTokenizer, AutoConfig, AutoFeatureExtractor, AutoProcessor, AutoImageProcessor, AutoModelForImageClassification, AutoModelForAudioClassification, AutoModelForVideoClassification

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Hardware detection
HAS_CUDA = torch.cuda.is_available()
HAS_MPS = hasattr(torch, "mps") and hasattr(torch.mps, "is_available") and torch.mps.is_available()
HAS_ROCM = (HAS_CUDA and hasattr(torch, '_C') and hasattr(torch._C, '_rocm_version')) or ('ROCM_HOME' in os.environ)
HAS_OPENVINO = False
try:
    import openvino
    HAS_OPENVINO = True
except ImportError:
    pass

HAS_QNN = False
try:
    import qnn_wrapper
    HAS_QNN = True
except ImportError:
    try:
        import qti
        HAS_QNN = True
    except ImportError:
        pass

HAS_WEBNN = os.environ.get("WEBNN_AVAILABLE", "0") == "1"
HAS_WEBGPU = os.environ.get("WEBGPU_AVAILABLE", "0") == "1"

class {model_name.replace("-", "").replace("/", "").title()}Skill:
    """Skill for {model_name} model with hardware platform support."""
    
    def __init__(self, model_id="{resolved_model_id}", device=None):
        """Initialize the skill."""
        self.model_id = model_id
        self.device = device or self.get_default_device()
        self.tokenizer = None
        self.processor = None
        self.model = None
        self.modality = "{modality}"
        logger.info(f"Initialized {model_name} skill with device: {{self.device}}")
    
    def get_default_device(self):
        """Get the best available device based on hardware availability."""
        # Check for CUDA
        if HAS_CUDA:
            return "cuda"
        
        # Check for Apple Silicon
        if HAS_MPS:
            return "mps"
        
        # Check for ROCm (AMD)
        if HAS_ROCM:
            return "rocm"
        
        # Check for OpenVINO (Intel)
        if HAS_OPENVINO:
            return "openvino"
        
        # Check for Qualcomm QNN
        if HAS_QNN:
            return "qualcomm"
        
        # Check for WebNN
        if HAS_WEBNN:
            return "webnn"
        
        # Check for WebGPU
        if HAS_WEBGPU:
            return "webgpu"
        
        # Default to CPU
        return "cpu"
    
    def load_model(self):
        """Load the model and tokenizer based on modality."""
        if self.model is None:
            logger.info(f"Loading {{self.modality}} model: {{self.model_id}}")
            
            # Load appropriate tokenizer/processor and model based on modality
            if self.modality == "audio":
                self.processor = AutoFeatureExtractor.from_pretrained(self.model_id)
                self.model = AutoModelForAudioClassification.from_pretrained(self.model_id)
            elif self.modality == "vision":
                self.processor = AutoImageProcessor.from_pretrained(self.model_id)
                self.model = AutoModelForImageClassification.from_pretrained(self.model_id)
            elif self.modality == "multimodal":
                self.processor = AutoProcessor.from_pretrained(self.model_id)
                self.model = AutoModel.from_pretrained(self.model_id)
            elif self.modality == "video":
                self.processor = AutoProcessor.from_pretrained(self.model_id)
                self.model = AutoModelForVideoClassification.from_pretrained(self.model_id)
            else:
                # Default to text
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
                self.model = AutoModel.from_pretrained(self.model_id)
            
            # Move to device
            if self.device != "cpu":
                self.model = self.model.to(self.device)
                logger.info(f"Model loaded and moved to {{self.device}}")
            else:
                logger.info("Model loaded on CPU")
    
    def process(self, text):
        """Process the input text and return the output."""
        # Ensure model is loaded
        self.load_model()
        
        logger.info(f"Processing input with {{self.modality}} model")
        
        if self.modality == "text":
            # Tokenize
            inputs = self.tokenizer(text, return_tensors="pt")
            
            # Move to device
            if self.device != "cpu":
                inputs = {{k: v.to(self.device) for k, v in inputs.items()}}
            
            # Run inference
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Convert to numpy for consistent output
            last_hidden_state = outputs.last_hidden_state.cpu().numpy()
            
            # Return formatted results
            return {{
                "model": self.model_id,
                "device": self.device,
                "last_hidden_state_shape": last_hidden_state.shape,
                "embedding": last_hidden_state.mean(axis=1).tolist(),
            }}
        elif self.modality in ["vision", "image"]:
            # Process image input (assuming np.array)
            inputs = self.processor(text, return_tensors="pt")
            
            # Move to device
            if self.device != "cpu":
                inputs = {{k: v.to(self.device) for k, v in inputs.items()}}
            
            # Run inference
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Process outputs based on model type
            if hasattr(outputs, "logits"):
                result = outputs.logits.cpu().numpy()
            else:
                result = outputs.last_hidden_state.cpu().numpy()
            
            return {{
                "model": self.model_id,
                "device": self.device,
                "shape": result.shape,
                "output": result.tolist() if result.size < 1000 else result.mean(axis=1).tolist(),
            }}
        else:
            # Generic processing for other modalities
            logger.info(f"Using generic processing for {{self.modality}} modality")
            return {{
                "model": self.model_id,
                "device": self.device,
                "modality": self.modality,
                "error": f"Direct processing for {{self.modality}} not implemented, use specific methods instead"
            }}
    
    def get_supported_hardware(self):
        """Get the list of supported hardware platforms."""
        supported = ["cpu"]
        
        if HAS_CUDA:
            supported.append("cuda")
        if HAS_MPS:
            supported.append("mps")
        if HAS_ROCM:
            supported.append("rocm")
        if HAS_OPENVINO:
            supported.append("openvino")
        if HAS_QNN:
            supported.append("qualcomm")
        if HAS_WEBNN:
            supported.append("webnn")
        if HAS_WEBGPU:
            supported.append("webgpu")
        
        return supported

# Factory function to create skill instance
def create_skill(model_id="{model_name}", device=None):
    """Create a skill instance."""
    return {model_name.replace("-", "").replace("/", "").title()}Skill(model_id=model_id, device=device)
''')
            
            logger.info(f"Generated skill file using default template: {output_path}")
            return {
                "success": True,
                "output_path": output_path,
                "model_name": model_name,
                "model_id": resolved_model_id,
                "template_from_db": False,
                "supported_platforms": ["cpu", "cuda", "mps", "rocm", "openvino", "qualcomm", "webnn", "webgpu"]
            }
        except Exception as e:
            logger.error(f"Error generating skill file: {e}")
            return {
                "success": False,
                "error": str(e),
                "model_name": model_name
            }
    except Exception as e:
        logger.error(f"Error generating skill file: {e}")
        return {
            "success": False,
            "error": str(e),
            "model_name": model_name
        }

def detect_available_hardware():
    """Detect available hardware platforms."""
    return get_available_hardware()

def generate_template_database_info(db_path: str = None):
    """Generate information about templates in the database."""
    if not HAS_TEMPLATE_DB:
        return "Template database not available. Install hardware_test_templates first."
    
    try:
        db = TemplateDatabase(db_path=db_path)
        templates = db.list_templates()
        
        result = []
        result.append(f"Found {len(templates)} templates in database:")
        
        # Group by model family
        families = {}
        for template in templates:
            family = template.get('model_family', 'unknown')
            if family not in families:
                families[family] = []
            families[family].append(template)
        
        # Output templates by family
        for family, templates in families.items():
            result.append(f"\n{family.upper()} ({len(templates)} templates):")
            for template in templates:
                result.append(f"  - {template['model_id']} ({template.get('modality', 'unknown')})")
        
        return "\n".join(result)
    except Exception as e:
        return f"Error accessing template database: {e}"

def main():
    parser = argparse.ArgumentParser(description="Integrated Skillset Generator - Enhanced Version")
    parser.add_argument("--model", "-m", type=str, help="Model to generate skill for")
    parser.add_argument("--platform", "-p", type=str, default="all", help="Platform to support (comma-separated or 'all')")
    parser.add_argument("--output-dir", "-o", type=str, help="Output directory for generated files")
    parser.add_argument("--use-db-templates", "-t", action="store_true", help="Use templates from the database")
    parser.add_argument("--db-path", type=str, help="Path to template database")
    parser.add_argument("--list-templates", "-l", action="store_true", help="List available templates in the database")
    parser.add_argument("--all-models", "-a", action="store_true", help="Generate skills for all models in MODEL_REGISTRY")
    parser.add_argument("--cross-platform", "-c", action="store_true", help="Generate skill with cross-platform support")
    parser.add_argument("--detect-hardware", "-d", action="store_true", help="Detect available hardware platforms")
    parser.add_argument("--family", "-f", type=str, help="Generate skills for all models in a specific family")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Set verbose logging if requested
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # List templates
    if args.list_templates:
        print(generate_template_database_info(args.db_path))
        return 0
    
    # Detect hardware
    if args.detect_hardware:
        hardware = detect_available_hardware()
        print(f"Available hardware platforms: {', '.join(hardware)}")
        return 0
    
    # Generate for all models
    if args.all_models:
        generated_files = []
        failures = []
        
        for model_name in MODEL_REGISTRY.keys():
            result = generate_skill_file(
                model_name, 
                args.platform, 
                args.output_dir, 
                use_db_templates=args.use_db_templates,
                db_path=args.db_path,
                cross_platform=args.cross_platform,
                verbose=args.verbose
            )
            
            if result.get("success", False):
                generated_files.append(result.get("output_path"))
            else:
                failures.append((model_name, result.get("error", "Unknown error")))
        
        print(f"Generated {len(generated_files)} skill files")
        if failures:
            print(f"Failed to generate {len(failures)} skill files:")
            for model_name, error in failures:
                print(f"  - {model_name}: {error}")
        
        return 0
    
    # Generate for a specific family
    if args.family:
        # Explicit family mappings to avoid errors
        family_models = []
        
        # Define family mappings for common model types
        family_map = {
            "text": ["bert", "bert-tiny", "t5", "t5-tiny", "llama", "llama-small", "qwen", "qwen2"],
            "text_embedding": ["bert", "bert-tiny"],
            "text_generation": ["t5", "t5-tiny", "llama", "llama-small", "qwen", "qwen2"],
            "vision": ["vit", "vit-tiny", "detr"],
            "audio": ["whisper", "whisper-small", "wav2vec2", "wav2vec2-small", "clap"],
            "multimodal": ["clip", "clip-small"],
            "vision_language": ["llava", "llava-small", "llava-tiny", "llava_next"]
        }
        
        # Try explicit mapping first
        if args.family.lower() in family_map:
            family_models = family_map[args.family.lower()]
        # Fallback to searching
        else:
            for model_name in MODEL_REGISTRY.keys():
                try:
                    model_family = get_model_family(model_name)
                    if args.family.lower() in model_family.lower():
                        family_models.append(model_name)
                except Exception as e:
                    logger.warning(f"Error determining family for {model_name}: {e}")
                    
        if not family_models:
            print(f"No models found for family: {args.family}")
            return 1
        
        print(f"Generating skill files for {len(family_models)} models in family '{args.family}'")
        
        generated_files = []
        failures = []
        
        for model_name in family_models:
            try:
                result = generate_skill_file(
                    model_name, 
                    args.platform, 
                    args.output_dir, 
                    use_db_templates=args.use_db_templates,
                    db_path=args.db_path,
                    cross_platform=args.cross_platform,
                    verbose=args.verbose
                )
                
                if result.get("success", False):
                    generated_files.append(result.get("output_path"))
                else:
                    failures.append((model_name, result.get("error", "Unknown error")))
            except Exception as e:
                failures.append((model_name, str(e)))
        
        print(f"Generated {len(generated_files)} skill files")
        if failures:
            print(f"Failed to generate {len(failures)} skill files:")
            for model_name, error in failures:
                print(f"  - {model_name}: {error}")
        
        return 0
    
    # Generate for a specific model
    if args.model:
        result = generate_skill_file(
            args.model, 
            args.platform, 
            args.output_dir, 
            use_db_templates=args.use_db_templates,
            db_path=args.db_path,
            cross_platform=args.cross_platform,
            verbose=args.verbose
        )
        
        if result.get("success", False):
            print(f"Generated skill file: {result.get('output_path')}")
            print(f"Model ID: {result.get('model_id')}")
            print(f"Used database template: {result.get('template_from_db', False)}")
            print(f"Supported platforms: {', '.join(result.get('supported_platforms', []))}")
        else:
            print(f"Failed to generate skill for {args.model}: {result.get('error', 'Unknown error')}")
            return 1
        
        return 0
    
    # If no action specified, print help
    parser.print_help()
    return 1

if __name__ == "__main__":
    sys.exit(main())