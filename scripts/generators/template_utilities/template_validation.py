#!/usr/bin/env python3
"""
Template validation utilities.

This module provides utilities for validating templates, including:
- Syntax validation: Ensure templates have valid Python syntax
- Hardware platform validation: Check hardware-specific code in templates
- Placeholder validation: Verify that required placeholders are present
- Schema validation: Verify database schema integrity
"""

import re
import json
import logging
from typing import Dict, Any, List, Tuple, Set, Optional

from .placeholder_helpers import extract_placeholders, get_standard_placeholders

logger = logging.getLogger(__name__)

# Hardware platform definitions
HARDWARE_PLATFORMS = [
    "cpu", "cuda", "rocm", "mps", "openvino", "qualcomm", "samsung", "webnn", "webgpu"
]

def validate_template_syntax(template: str) -> Tuple[bool, List[str]]:
    """
    Validate template syntax (balanced braces, valid Python syntax, etc.)
    
    Args:
        template (str): The template string to validate
        
    Returns:
        Tuple[bool, List[str]]: Success status and list of errors
    """
    errors = []
    
    # Check for balanced braces in placeholders
    if template.count('{') != template.count('}'):
        errors.append("Unbalanced braces in template")
    
    # Check for Python syntax errors
    try:
        # We need to replace all placeholder patterns with actual values for compilation
        placeholders = extract_placeholders(template)
        test_template = template
        
        for placeholder in placeholders:
            test_template = test_template.replace(f"{{{placeholder}}}", f'"{placeholder}"')
        
        # Try to compile the template as Python code
        compile(test_template, '<template>', 'exec')
    except SyntaxError as e:
        errors.append(f"Python syntax error: {e}")
    
    # Check for common template issues
    if "{{" in template or "}}" in template:
        errors.append("Double braces detected: {{ or }} should be single { or }")
    
    if "\\n" in template and '"""' in template:
        # This could be legitimate in some cases, so just add a warning
        errors.append("Warning: \\n escape sequence found in triple-quoted string")
    
    return len(errors) == 0, errors

def validate_hardware_support(template: str, hardware_platform: Optional[str] = None) -> Tuple[bool, Dict[str, bool]]:
    """
    Validate hardware support in a template
    
    Args:
        template (str): The template string to validate
        hardware_platform (Optional[str]): Specific hardware platform to validate for
        
    Returns:
        Tuple[bool, Dict[str, bool]]: Success status and hardware support dictionary
    """
    # Initialize hardware support status for all platforms
    hardware_support = {platform: False for platform in HARDWARE_PLATFORMS}
    hardware_support['cpu'] = True  # CPU support is assumed for all templates
    
    # Hardware-specific imports and keywords to check for
    hardware_patterns = {
        'cuda': [
            r'torch\.cuda', r'device\s*=\s*[\'"]cuda[\'"]', r'\.to\([\'"]cuda[\'"]',
            r'\.cuda\(', r'CUDA', r'nvidia', r'cuDNN'
        ],
        'rocm': [
            r'rocm', r'AMD', r'amd', r'ROCm', r'hip', r'HIP', r'AMD_GPU'
        ],
        'mps': [
            r'mps', r'torch\.backends\.mps', r'device\s*=\s*[\'"]mps[\'"]',
            r'\.to\([\'"]mps[\'"]', r'Apple', r'apple', r'metal', r'Metal'
        ],
        'openvino': [
            r'openvino', r'OpenVINO', r'OPENVINO', r'intel', r'Intel', r'INTEL',
            r'IE_PLUGINS', r'ie_plugins', r'ie_core', r'IE_CORE'
        ],
        'qualcomm': [
            r'qualcomm', r'Qualcomm', r'QUALCOMM', r'QNN', r'qnn', r'Hexagon',
            r'hexagon', r'Snapdragon', r'snapdragon'
        ],
        'samsung': [
            r'samsung', r'Samsung', r'SAMSUNG', r'exynos', r'Exynos', r'EXYNOS',
            r'NPU', r'npu', r'enn'
        ],
        'webnn': [
            r'webnn', r'WebNN', r'WEBNN', r'navigator\.ml', r'navigator_ml',
            r'transformers_js', r'ml_graph', r'MLGraph'
        ],
        'webgpu': [
            r'webgpu', r'WebGPU', r'WEBGPU', r'GPUDevice', r'gpudevice',
            r'transformers_js', r'wgsl', r'WGSL', r'shader'
        ]
    }
    
    # Check for hardware-specific patterns in the template
    for platform, patterns in hardware_patterns.items():
        for pattern in patterns:
            if re.search(pattern, template):
                hardware_support[platform] = True
                break
    
    # If a specific hardware platform is specified, check if it's supported
    if hardware_platform:
        return hardware_support.get(hardware_platform, False), hardware_support
    
    # Otherwise, return overall validation status and hardware support dict
    return True, hardware_support

def validate_placeholders_in_template(template: str) -> Tuple[bool, List[str], List[str]]:
    """
    Validate placeholders in a template
    
    Args:
        template (str): The template string to validate
        
    Returns:
        Tuple[bool, List[str], List[str]]: Success status, missing required placeholders, and all placeholders
    """
    # Extract all placeholders from the template
    placeholders = extract_placeholders(template)
    
    # Check for mandatory placeholders based on template type
    standard_placeholders = get_standard_placeholders()
    mandatory_placeholders = {
        name for name, info in standard_placeholders.items() 
        if info.get('required', False)
    }
    
    # Find missing mandatory placeholders
    missing_mandatory = mandatory_placeholders - placeholders
    
    return len(missing_mandatory) == 0, list(missing_mandatory), list(placeholders)

def validate_template(template: str, template_type: str, model_type: str, hardware_platform: Optional[str] = None) -> Tuple[bool, Dict[str, Any]]:
    """
    Comprehensive template validation (syntax, hardware support, placeholders)
    
    Args:
        template (str): The template string to validate
        template_type (str): The type of template (test, benchmark, skill)
        model_type (str): The model type the template is for
        hardware_platform (Optional[str]): Specific hardware platform to validate for
        
    Returns:
        Tuple[bool, Dict[str, Any]]: Success status and validation results dictionary
    """
    validation_results = {
        'syntax': {'success': False, 'errors': []},
        'hardware': {'success': False, 'support': {}},
        'placeholders': {'success': False, 'missing': [], 'all': []}
    }
    
    # Validate syntax
    syntax_valid, syntax_errors = validate_template_syntax(template)
    validation_results['syntax']['success'] = syntax_valid
    validation_results['syntax']['errors'] = syntax_errors
    
    # Validate hardware support
    hardware_valid, hardware_support = validate_hardware_support(template, hardware_platform)
    validation_results['hardware']['success'] = hardware_valid
    validation_results['hardware']['support'] = hardware_support
    
    # Validate placeholders
    placeholders_valid, missing_placeholders, all_placeholders = validate_placeholders_in_template(template)
    validation_results['placeholders']['success'] = placeholders_valid
    validation_results['placeholders']['missing'] = missing_placeholders
    validation_results['placeholders']['all'] = all_placeholders
    
    # Determine overall validation status
    validation_success = syntax_valid and hardware_valid and placeholders_valid
    
    return validation_success, validation_results

def validate_template_db_schema(columns: List[str]) -> Tuple[bool, List[str]]:
    """
    Validate template database schema columns
    
    Args:
        columns (List[str]): List of column names in the templates table
        
    Returns:
        Tuple[bool, List[str]]: Success status and list of missing required columns
    """
    required_columns = [
        'model_type', 'template_type', 'template', 'hardware_platform', 
        'validation_status', 'parent_template', 'modality', 'last_updated'
    ]
    
    missing_columns = [col for col in required_columns if col not in columns]
    
    return len(missing_columns) == 0, missing_columns