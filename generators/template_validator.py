#!/usr/bin/env python
# Template validation system for IPFS Accelerate
# Validates templates for syntax correctness and compatibility
# Interfaces with DuckDB template database for template storage and retrieval

import os
import sys
import logging
import importlib
import tempfile
import argparse
import ast
import re
import json
from typing import Dict, List, Any, Tuple, Optional, Set
from pathlib import Path
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Try importing template inheritance system
try:
    # First try the fixed version
    try:
        from template_inheritance_system_fixed import (
            register_template, register_template_directory,
            validate_all_templates, test_template_compatibility,
            get_template_inheritance_graph, TemplateError
        )
        template_system_available = True
        logger.info("Using fixed template inheritance system")
    except ImportError:
        # Fall back to original if fixed version not available
        from template_inheritance_system import (
            register_template, register_template_directory,
            validate_all_templates, test_template_compatibility,
            get_template_inheritance_graph, TemplateError
        )
        template_system_available = True
        logger.info("Using original template inheritance system")
except ImportError:
    logger.warning("Template inheritance system not available, using basic verification")
    template_system_available = False

# Try importing template database
try:
    from hardware_test_templates.template_database import TemplateDatabase
    template_db_available = True
except ImportError:
    logger.warning("Template database not available, using file-based validation only")
    template_db_available = False

def validate_python_syntax(content: str) -> Tuple[bool, List[str]]:
    """
    Validate Python syntax in the template
    
    Args:
        content: Template content as string
        
    Returns:
        Tuple of (valid, list of errors)
    """
    errors = []
    try:
        ast.parse(content)
        return True, []
    except SyntaxError as e:
        errors.append(f"Syntax error at line {e.lineno}: {e.msg}")
        return False, errors

def validate_imports(content: str) -> Tuple[bool, List[str]]:
    """
    Validate imports in the template
    
    Args:
        content: Template content as string
        
    Returns:
        Tuple of (valid, list of errors)
    """
    errors = []
    required_imports = {'os', 'sys', 'torch', 'logging'}
    found_imports = set()
    
    # Find all import statements
    import_pattern = re.compile(r'import\s+([\w\.]+)|from\s+([\w\.]+)\s+import')
    for match in import_pattern.finditer(content):
        if match.group(1):
            # 'import x' form
            module = match.group(1).split('.')[0]
            found_imports.add(module)
        elif match.group(2):
            # 'from x import y' form
            module = match.group(2).split('.')[0]
            found_imports.add(module)
    
    # Check for missing required imports
    missing_imports = required_imports - found_imports
    if missing_imports:
        errors.append(f"Missing required imports: {', '.join(missing_imports)}")
    
    return len(errors) == 0, errors

def validate_class_structure(content: str) -> Tuple[bool, List[str]]:
    """
    Validate class structure in the template
    
    Args:
        content: Template content as string
        
    Returns:
        Tuple of (valid, list of errors)
    """
    errors = []
    
    # Parse the AST to analyze structure
    try:
        tree = ast.parse(content)
        
        # Find class definitions
        classes = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
        
        if not classes:
            errors.append("No class definitions found in template")
            return False, errors
        
        # Check for test class naming convention
        test_classes = [cls for cls in classes if cls.name.startswith('Test')]
        if not test_classes:
            errors.append("No test classes found (class names should start with 'Test')")
        
        # Check for required methods in test classes
        for cls in test_classes:
            methods = [node.name for node in cls.body if isinstance(node, ast.FunctionDef)]
            
            # Check for setup_class method
            if 'setup_class' not in methods:
                errors.append(f"Class {cls.name} missing required method: setup_class")
            
            # Check for at least one test method
            test_methods = [m for m in methods if m.startswith('test_')]
            if not test_methods:
                errors.append(f"Class {cls.name} has no test methods (should start with 'test_')")
            
            # Check for teardown_class method
            if 'teardown_class' not in methods:
                errors.append(f"Class {cls.name} missing recommended method: teardown_class")
        
        return len(errors) == 0, errors
    except SyntaxError:
        # This should be caught by validate_python_syntax
        errors.append("Cannot analyze class structure due to syntax errors")
        return False, errors

def validate_resource_pool_usage(content: str) -> Tuple[bool, List[str]]:
    """
    Validate ResourcePool usage in the template
    
    Args:
        content: Template content as string
        
    Returns:
        Tuple of (valid, list of errors)
    """
    errors = []
    warnings = []
    
    # Check for ResourcePool import
    if 'resource_pool' not in content or 'get_global_resource_pool' not in content:
        errors.append("Template does not use ResourcePool (missing import or usage)")
        return False, errors
    
    # Check for common ResourcePool usage patterns
    if 'pool.get_model' not in content:
        warnings.append("Template may not be using ResourcePool for model caching")
    
    if 'pool.get_resource' not in content:
        warnings.append("Template may not be using ResourcePool for resource sharing")
    
    # Add warnings to errors list
    if warnings:
        errors.extend([f"WARNING: {w}" for w in warnings])
    
    # Determine success based only on critical errors (not warnings)
    success = not any(not e.startswith("WARNING:") for e in errors)
    return success, errors

def validate_hardware_awareness(content: str) -> Tuple[bool, List[str]]:
    """
    Validate hardware awareness in the template
    
    Args:
        content: Template content as string
        
    Returns:
        Tuple of (valid, list of errors)
    """
    errors = []
    
    # Check for device handling
    if 'device' not in content.lower() or 'cuda' not in content.lower():
        errors.append("Template does not appear to handle different hardware devices")
    
    # Check for basic CUDA availability check
    if 'torch.cuda.is_available()' not in content:
        errors.append("Missing CUDA availability check")
    
    # Check for device allocation
    if '.to(device)' not in content and '.to("cuda")' not in content:
        errors.append("No device allocation found for models/tensors")
    
    return len(errors) == 0, errors

def validate_cross_platform_support(content: str) -> Tuple[bool, List[str], List[str]]:
    """
    Validate cross-platform support in the template
    
    Args:
        content: Template content as string
        
    Returns:
        Tuple of (valid, list of errors, list of supported platforms)
    """
    errors = []
    warnings = []
    
    # Define hardware platforms to check for with more extensive patterns
    hardware_platforms = [
        ('cuda', r'(cuda|gpu)'),
        ('cpu', r'cpu'),
        ('mps', r'(mps|apple|m1|m2)'),
        ('rocm', r'(rocm|amd)'),
        ('openvino', r'(openvino|intel)'),
        ('qualcomm', r'(qualcomm|qnn|hexagon)'),
        ('webnn', r'webnn'),
        ('webgpu', r'webgpu'),
        ('samsung', r'samsung'),
        ('mediatek', r'mediatek')
    ]
    
    supported_platforms = []
    
    # More sophisticated hardware detection checks
    hardware_checks = {
        'cuda': [
            r'HAS_CUDA', 
            r'torch\.cuda\.is_available', 
            r'device\s*=\s*[\'"]cuda[\'"]',
            r'torch\.device\([\'"]cuda[\'"]',
            r'\.to\([\'"]cuda[\'"]',
            r'cuda_version'
        ],
        'cpu': [
            r'device\s*=\s*[\'"]cpu[\'"]',
            r'torch\.device\([\'"]cpu[\'"]',
            r'\.to\([\'"]cpu[\'"]',
            r'HAS_CPU',
            r'USE_CPU'
        ],
        'mps': [
            r'HAS_MPS', 
            r'torch\.mps\.is_available', 
            r'device\s*=\s*[\'"]mps[\'"]',
            r'torch\.device\([\'"]mps[\'"]',
            r'\.to\([\'"]mps[\'"]',
            r'APPLE_SILICON'
        ],
        'rocm': [
            r'HAS_ROCM', 
            r'_rocm_version', 
            r'ROCM_HOME',
            r'AMD_COMPATIBLE',
            r'amd_compatible',
            r'HIP_VERSION'
        ],
        'openvino': [
            r'HAS_OPENVINO',
            r'openvino',
            r'import\s+openvino',
            r'INTEL_HARDWARE',
            r'openvino\.runtime',
            r'ov_model',
            r'optimum\.intel'
        ],
        'qualcomm': [
            r'HAS_QUALCOMM',
            r'QUALCOMM_SDK',
            r'qnn_wrapper',
            r'import\s+qti',
            r'QNN_RUNTIME',
            r'hexagon',
            r'snapdragon'
        ],
        'webnn': [
            r'HAS_WEBNN',
            r'WEBNN_AVAILABLE',
            r'WEBNN_SIMULATION',
            r'webnn_tensor',
            r'browser="edge"',
            r'OVModelForWebNN'
        ],
        'webgpu': [
            r'HAS_WEBGPU',
            r'WEBGPU_AVAILABLE',
            r'WEBGPU_SIMULATION',
            r'webgpu_tensor',
            r'WebGPUCompute',
            r'shader',
            r'compute_shader',
            r'webgpu_context'
        ],
        'samsung': [
            r'HAS_SAMSUNG',
            r'SAMSUNG_NPU',
            r'exynos',
            r'EXYNOS_VERSION',
            r'samsung_runtime'
        ],
        'mediatek': [
            r'HAS_MEDIATEK',
            r'MEDIATEK_APU',
            r'dimensity',
            r'neuron_adapter',
            r'apu_runtime'
        ]
    }
    
    # Enhanced check for modern template feature: resource pool
    resource_pool_aware = (
        'resource_pool' in content and 
        ('get_resource_pool' in content or 'ResourcePool' in content)
    )
    
    # Check for hardware detection via hardware_detection module (preferred method)
    uses_hardware_detection_module = (
        'hardware_detection' in content and 
        ('detect_available_hardware' in content or 'get_available_hardware' in content)
    )
    
    # New check for template inheritance
    uses_template_inheritance = (
        'template_inheritance' in content or
        'BaseTemplate' in content or
        'inherit' in content and ('template_base' in content or 'template_parent' in content)
    )
    
    # Check for centralized hardware detection
    uses_central_detection = uses_hardware_detection_module or 'detect_hardware' in content
    if uses_central_detection:
        # Template likely uses the centralized hardware detection system
        logger.debug("Found centralized hardware detection")
        
        # When centralized detection is used, we need to look for hardware-specific code
        # that would actually use the detected hardware
        for platform_name, patterns in hardware_checks.items():
            for pattern in patterns:
                if re.search(pattern, content):
                    if platform_name not in supported_platforms:
                        supported_platforms.append(platform_name)
                    break
                    
        # If template uses centralized detection but we couldn't find specific platforms,
        # assume it handles all major platforms
        if len(supported_platforms) < 3 and uses_hardware_detection_module:
            supported_platforms = ['cpu', 'cuda', 'mps', 'rocm', 'openvino', 'qualcomm', 'webnn', 'webgpu']
            logger.debug("Template uses hardware_detection module, assuming support for all platforms")
    else:
        # Check for explicit hardware checks
        for platform_name, patterns in hardware_checks.items():
            for pattern in patterns:
                if re.search(pattern, content):
                    if platform_name not in supported_platforms:
                        supported_platforms.append(platform_name)
                    break
        
        # If still not found hardware, check for mentions using generic patterns
        if not supported_platforms:
            for platform_name, pattern in hardware_platforms:
                if re.search(pattern, content, re.IGNORECASE):
                    supported_platforms.append(platform_name)
    
    # If template uses resource pool, it likely inherits hardware support from the pool
    if resource_pool_aware and len(supported_platforms) < 3:
        logger.debug("Template uses resource pool, assuming broader hardware support")
        # Add core platforms if missing but template uses resource pool
        for platform in ['cpu', 'cuda']:
            if platform not in supported_platforms:
                supported_platforms.append(platform)
                
    # If template has template inheritance, check for base template references
    if uses_template_inheritance and len(supported_platforms) < 3:
        logger.debug("Template uses inheritance, hardware support might come from parent templates")
        warnings.append("Template uses inheritance; hardware support may be defined in parent templates")
    
    # Core platforms that should be supported
    core_platforms = {'cuda', 'cpu'}
    missing_core = core_platforms - set(supported_platforms)
    
    if missing_core:
        errors.append(f"Missing support for core hardware platforms: {', '.join(missing_core)}")
    
    # Recommended platforms
    recommended_platforms = {'mps', 'rocm', 'openvino'}
    missing_recommended = recommended_platforms - set(supported_platforms)
    
    if missing_recommended:
        warnings.append(f"Missing recommended hardware platforms: {', '.join(missing_recommended)}")
    
    # Web platforms
    web_platforms = {'webnn', 'webgpu'}
    has_web = any(p in supported_platforms for p in web_platforms)
    
    if not has_web:
        warnings.append("No web platform support detected (WebNN/WebGPU)")
    
    # Check for mobile AI hardware support (Qualcomm, Samsung, MediaTek)
    mobile_platforms = {'qualcomm', 'samsung', 'mediatek'}
    has_mobile = any(p in supported_platforms for p in mobile_platforms)
    
    if not has_mobile:
        warnings.append("No mobile AI hardware support detected (Qualcomm, Samsung, MediaTek)")
    
    # Check for all platforms
    if len(supported_platforms) < 3:
        warnings.append(f"Limited hardware support detected, only found: {', '.join(supported_platforms)}")
    
    # Add warnings to errors
    if warnings:
        errors.extend([f"WARNING: {w}" for w in warnings])
    
    # Only count serious errors (not warnings) for validity
    success = not any(not e.startswith("WARNING:") for e in errors)
    
    return success, errors, supported_platforms

def validate_template_variables(content: str) -> Tuple[bool, List[str]]:
    """
    Validate template variables in the template
    
    Args:
        content: Template content as string
        
    Returns:
        Tuple of (valid, list of errors)
    """
    errors = []
    
    # Check for template variables
    template_vars = re.findall(r'{{(.*?)}}', content)
    
    # Common required variables
    required_vars = ['model_name', 'model_class']
    
    # Check if required variables are present
    found_vars = [var.strip() for var in template_vars]
    
    # Extract variable names from more complex expressions
    cleaned_vars = []
    for var in found_vars:
        # Handle expressions like model_name.replace("-", "")
        if '.' in var:
            base_var = var.split('.')[0].strip()
            cleaned_vars.append(base_var)
        else:
            cleaned_vars.append(var)
    
    # Find missing required variables
    missing_vars = []
    for req_var in required_vars:
        if req_var not in cleaned_vars:
            missing_vars.append(req_var)
    
    if missing_vars:
        errors.append(f"Missing required template variables: {', '.join(missing_vars)}")
    
    # Verify variable patterns are valid
    invalid_vars = []
    for var in template_vars:
        # Check for common errors in variable expressions
        if var.count('{') != var.count('}'):
            invalid_vars.append(var)
        elif var.count('(') != var.count(')'):
            invalid_vars.append(var)
    
    if invalid_vars:
        errors.append(f"Invalid template variable syntax: {', '.join(invalid_vars)}")
    
    return len(errors) == 0, errors

def validate_template_content(content: str) -> Dict[str, Any]:
    """
    Validate template content with multiple validation rules
    
    Args:
        content: Template content as string
        
    Returns:
        Dictionary with validation results
    """
    # Run all validators
    validators = [
        ('syntax', validate_python_syntax),
        ('imports', validate_imports),
        ('class_structure', validate_class_structure),
        ('resource_pool', validate_resource_pool_usage),
        ('hardware_awareness', validate_hardware_awareness),
        ('template_variables', validate_template_variables)
    ]
    
    all_valid = True
    all_errors = []
    results_by_validator = {}
    
    for validator_name, validator_func in validators:
        valid, errors = validator_func(content)
        results_by_validator[validator_name] = {
            'valid': valid,
            'errors': errors
        }
        
        all_valid = all_valid and valid
        all_errors.extend([f"{validator_name}: {error}" for error in errors])
    
    # Run cross-platform check separately to capture supported platforms
    cp_valid, cp_errors, supported_platforms = validate_cross_platform_support(content)
    results_by_validator['cross_platform'] = {
        'valid': cp_valid,
        'errors': cp_errors,
        'supported_platforms': supported_platforms
    }
    all_valid = all_valid and cp_valid
    all_errors.extend([f"cross_platform: {error}" for error in cp_errors])
    
    # If template inheritance system is available, use it for additional validation
    template_inheritance_results = None
    if template_system_available:
        try:
            with tempfile.NamedTemporaryFile(suffix='.py', delete=False) as temp_file:
                temp_path = temp_file.name
                temp_file.write(content.encode('utf-8'))
            
            # Register and validate with template system
            try:
                template = register_template(temp_path, os.path.basename(temp_path))
                valid, errors = template.validate()
                template_inheritance_results = {
                    'valid': valid,
                    'errors': errors
                }
                
                all_valid = all_valid and valid
                all_errors.extend([f"template_system: {error}" for error in errors])
            except TemplateError as e:
                template_inheritance_results = {
                    'valid': False,
                    'errors': [str(e)]
                }
                all_valid = False
                all_errors.append(f"template_system: {str(e)}")
            
            # Clean up temporary file
            os.unlink(temp_path)
        except Exception as e:
            logger.error(f"Error using template inheritance system: {str(e)}")
    
    # Combine all results
    result = {
        'valid': all_valid,
        'errors': all_errors,
        'validators': results_by_validator,
        'supported_platforms': supported_platforms
    }
    
    if template_inheritance_results:
        result['template_inheritance'] = template_inheritance_results
    
    return result

def validate_template_file(file_path: str) -> Dict[str, Any]:
    """
    Validate a template file with multiple validation rules
    
    Args:
        file_path: Path to the template file
        
    Returns:
        Dictionary with validation results
    """
    logger.info(f"Validating template file: {file_path}")
    
    # Read template content
    try:
        with open(file_path, 'r') as f:
            content = f.read()
    except (IOError, UnicodeDecodeError) as e:
        return {
            'valid': False,
            'file': file_path,
            'errors': [f"Failed to read file: {str(e)}"]
        }
    
    result = validate_template_content(content)
    result['file'] = file_path
    
    return result

def validate_template_from_db(model_id: str, db_path: str = None) -> Dict[str, Any]:
    """
    Validate a template from the database
    
    Args:
        model_id: Model ID to validate
        db_path: Optional path to database
        
    Returns:
        Dictionary with validation results
    """
    if not template_db_available:
        return {
            'valid': False,
            'model_id': model_id,
            'errors': ["Template database not available"]
        }
    
    # Initialize database
    db = TemplateDatabase(db_path=db_path) if db_path else TemplateDatabase()
    
    # Get template content
    template_data = db.get_template_with_metadata(model_id)
    if not template_data:
        return {
            'valid': False,
            'model_id': model_id,
            'errors': [f"Template not found for model_id: {model_id}"]
        }
    
    content = template_data['template_content']
    
    logger.info(f"Validating template from database: {model_id}")
    
    # Validate template content
    result = validate_template_content(content)
    
    # Add model metadata
    result['model_id'] = model_id
    result['model_name'] = template_data.get('model_name')
    result['model_family'] = template_data.get('model_family')
    result['modality'] = template_data.get('modality')
    result['last_updated'] = template_data.get('last_updated')
    
    return result

def validate_generator_compatibility(template_content: str, generator_type: str) -> Tuple[bool, List[str]]:
    """
    Validate template compatibility with specific generator types
    
    Args:
        template_content: Template content as string
        generator_type: Type of generator ('merged', 'integrated_skillset', 'fixed', 'simple')
        
    Returns:
        Tuple of (valid, list of errors)
    """
    errors = []
    
    # Check generator-specific requirements
    if generator_type == 'merged':
        # Merged generator requirements
        if 'from merged_test_generator import' not in template_content and 'import merged_test_generator' not in template_content:
            errors.append("No merged_test_generator import found - template may not be compatible with merged generator")
            
        if 'generate_test' not in template_content:
            errors.append("No generate_test function found - required for merged generator compatibility")
            
    elif generator_type == 'integrated_skillset':
        # Integrated skillset generator requirements
        if 'from integrated_skillset_generator import' not in template_content and 'import integrated_skillset_generator' not in template_content:
            errors.append("No integrated_skillset_generator import found - template may not be compatible with integrated skillset generator")
            
        if 'generate_skillset' not in template_content:
            errors.append("No generate_skillset function found - required for integrated skillset generator")
            
    elif generator_type == 'fixed':
        # Fixed generator requirements
        if 'from fixed_merged_test_generator import' not in template_content and 'import fixed_merged_test_generator' not in template_content:
            errors.append("No fixed_merged_test_generator import found - template may not be compatible with fixed generator")
            
    elif generator_type == 'simple':
        # Simple generator requirements
        if 'from simple_test_generator import' not in template_content and 'import simple_test_generator' not in template_content:
            errors.append("No simple_test_generator import found - template may not be compatible with simple generator")
    
    return len(errors) == 0, errors

def validate_all_db_templates(db_path: str = None, 
                             model_family: str = None, 
                             modality: str = None,
                             generator_type: str = None) -> Dict[str, Dict[str, Any]]:
    """
    Validate all templates in the database
    
    Args:
        db_path: Optional path to database
        model_family: Optional filter by model family
        modality: Optional filter by modality
        generator_type: Optional filter by generator type ('merged', 'integrated_skillset', 'fixed', 'simple')
        
    Returns:
        Dictionary mapping model IDs to validation results
    """
    if not template_db_available:
        return {"error": "Template database not available"}
    
    # Initialize database
    db = TemplateDatabase(db_path=db_path) if db_path else TemplateDatabase()
    
    # Get templates matching criteria
    templates = db.list_templates(model_family=model_family, modality=modality)
    
    results = {}
    for template in templates:
        model_id = template['model_id']
        result = validate_template_from_db(model_id, db_path)
        
        # If generator_type specified, validate compatibility
        if generator_type and 'template_content' in result:
            generator_valid, generator_errors = validate_generator_compatibility(
                result['template_content'], generator_type
            )
            
            # Add generator compatibility results
            result['generator_compatibility'] = {
                'generator_type': generator_type,
                'valid': generator_valid,
                'errors': generator_errors
            }
            
            # Update overall validity
            result['valid'] = result['valid'] and generator_valid
            result['errors'].extend([f"generator_compatibility: {error}" for error in generator_errors])
        
        results[model_id] = result
    
    return results

def validate_template_directory(directory_path: str) -> Dict[str, Dict[str, Any]]:
    """
    Validate all templates in a directory
    
    Args:
        directory_path: Path to directory containing templates
        
    Returns:
        Dictionary mapping file names to validation results
    """
    results = {}
    
    # Find all Python files in the directory
    for file_path in Path(directory_path).glob('*.py'):
        # Skip files starting with underscore
        if file_path.name.startswith('_'):
            continue
        
        # Skip non-template files (basic check)
        if not (file_path.name.startswith('hf_') or file_path.name.endswith('template.py')):
            continue
        
        # Validate the template
        result = validate_template_file(str(file_path))
        results[file_path.name] = result
    
    return results

def test_compatibility(templates_dir: str) -> Dict[str, Dict[str, Any]]:
    """
    Test template compatibility with available hardware
    
    Args:
        templates_dir: Directory containing templates
        
    Returns:
        Dictionary of compatibility test results
    """
    if not template_system_available:
        return {"error": "Template inheritance system not available"}
    
    # Register all templates
    register_template_directory(templates_dir)
    
    # Test compatibility
    return test_template_compatibility()

def store_validation_results(results: Dict[str, Any], 
                            output_file: str = None,
                            store_in_db: bool = False,
                            db_path: str = None) -> bool:
    """
    Store validation results to a file or database
    
    Args:
        results: Validation results
        output_file: Path to output file (JSON)
        store_in_db: Whether to store in database
        db_path: Optional path to database
        
    Returns:
        True if successful, False otherwise
    """
    success = True
    
    # Store to file if specified
    if output_file:
        try:
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Validation results saved to: {output_file}")
        except Exception as e:
            logger.error(f"Failed to write validation results to file: {e}")
            success = False
    
    # Store to database if specified
    if store_in_db and template_db_available:
        try:
            # Initialize database
            db = TemplateDatabase(db_path=db_path) if db_path else TemplateDatabase()
            
            # Connect to database
            conn = db._get_connection()
            
            try:
                # Check if validation_results table exists
                table_exists = conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='validation_results'"
                ).fetchone()
                
                if not table_exists:
                    # Create validation results table
                    conn.execute("""
                    CREATE TABLE IF NOT EXISTS validation_results (
                        result_id INTEGER PRIMARY KEY,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        model_id VARCHAR,
                        valid BOOLEAN,
                        validators_json JSON,
                        error_count INTEGER,
                        supported_platforms_json JSON
                    )
                    """)
                
                # Prepare data for each model/template
                if 'model_id' in results:
                    # Single template result
                    models_results = [results]
                else:
                    # Multiple template results
                    models_results = list(results.values())
                
                # Store each result
                for result in models_results:
                    model_id = result.get('model_id') or os.path.basename(result.get('file', ''))
                    valid = result.get('valid', False)
                    error_count = len(result.get('errors', []))
                    validators_json = json.dumps(result.get('validators', {}))
                    supported_platforms = result.get('supported_platforms', [])
                    supported_platforms_json = json.dumps(supported_platforms)
                    
                    # Insert into database
                    conn.execute("""
                    INSERT INTO validation_results (
                        timestamp, model_id, valid, validators_json, error_count, supported_platforms_json
                    )
                    VALUES (?, ?, ?, ?, ?, ?)
                    """, [datetime.now(), model_id, valid, validators_json, error_count, supported_platforms_json])
                
                logger.info(f"Validation results stored in database for {len(models_results)} templates")
            finally:
                conn.close()
        except Exception as e:
            logger.error(f"Failed to store validation results in database: {e}")
            success = False
    
    return success

def get_validation_history(model_id: str, 
                          limit: int = 10, 
                          db_path: str = None) -> List[Dict[str, Any]]:
    """
    Get validation history for a model
    
    Args:
        model_id: ID of the model
        limit: Maximum number of results to return
        db_path: Optional path to database
        
    Returns:
        List of validation results ordered by timestamp
    """
    if not template_db_available:
        return [{"error": "Template database not available"}]
    
    # Initialize database
    db = TemplateDatabase(db_path=db_path) if db_path else TemplateDatabase()
    
    # Connect to database
    conn = db._get_connection()
    
    try:
        # Check if validation_results table exists
        table_exists = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='validation_results'"
        ).fetchone()
        
        if not table_exists:
            return []
        
        # Get validation history
        results = conn.execute("""
        SELECT
            result_id,
            timestamp,
            valid,
            error_count,
            validators_json,
            supported_platforms_json
        FROM
            validation_results
        WHERE
            model_id = ?
        ORDER BY
            timestamp DESC
        LIMIT ?
        """, [model_id, limit]).fetchdf()
        
        if results.empty:
            return []
        
        # Convert to list of dictionaries
        history = []
        for _, row in results.iterrows():
            # Parse JSON fields
            try:
                validators = json.loads(row['validators_json'])
            except:
                validators = {}
            
            try:
                supported_platforms = json.loads(row['supported_platforms_json'])
            except:
                supported_platforms = []
            
            history.append({
                'result_id': row['result_id'],
                'timestamp': row['timestamp'],
                'valid': bool(row['valid']),
                'error_count': row['error_count'],
                'validators': validators,
                'supported_platforms': supported_platforms
            })
        
        return history
    except Exception as e:
        logger.error(f"Error getting validation history: {e}")
        return []
    finally:
        conn.close()

def generate_validation_report(results: Dict[str, Dict[str, Any]]) -> str:
    """
    Generate a validation report in markdown format
    
    Args:
        results: Validation results
        
    Returns:
        Markdown report
    """
    # Count valid and invalid templates
    valid_count = sum(1 for r in results.values() if r.get('valid', False))
    invalid_count = len(results) - valid_count
    total_count = len(results)
    
    # Platform support stats
    platform_counts = {
        'cuda': 0,
        'cpu': 0,
        'mps': 0,
        'rocm': 0,
        'openvino': 0,
        'webnn': 0,
        'webgpu': 0
    }
    
    # Count supported platforms
    for result in results.values():
        for platform in result.get('supported_platforms', []):
            if platform in platform_counts:
                platform_counts[platform] += 1
    
    # Start building the report
    report = f"""# Template Validation Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary
- Total templates: {total_count}
- Valid templates: {valid_count} ({valid_count/total_count*100:.1f}%)
- Invalid templates: {invalid_count} ({invalid_count/total_count*100:.1f}%)

## Hardware Platform Support
| Platform | Count | Percentage |
|----------|-------|------------|
"""
    
    # Add platform support stats
    for platform, count in platform_counts.items():
        percentage = count/total_count*100 if total_count > 0 else 0
        report += f"| {platform} | {count} | {percentage:.1f}% |\n"
    
    # Add invalid templates section if any
    if invalid_count > 0:
        report += "\n## Invalid Templates\n"
        
        for template_name, result in results.items():
            if not result.get('valid', True):
                model_id = result.get('model_id', template_name)
                model_family = result.get('model_family', 'unknown')
                modality = result.get('modality', 'unknown')
                
                report += f"\n### {model_id} ({model_family}, {modality})\n\n"
                report += f"**Supported platforms**: {', '.join(result.get('supported_platforms', []))}\n\n"
                report += "**Errors:**\n"
                
                for error in result.get('errors', []):
                    report += f"- {error}\n"
    
    return report

def main():
    """Main function for standalone usage"""
    parser = argparse.ArgumentParser(description="Template validation system")
    parser.add_argument("--file", type=str, help="Validate a single template file")
    parser.add_argument("--dir", type=str, help="Validate all templates in a directory")
    parser.add_argument("--model-id", type=str, help="Validate a specific model ID from database")
    parser.add_argument("--family", type=str, help="Validate all templates in a model family")
    parser.add_argument("--modality", type=str, help="Validate all templates with a specific modality")
    parser.add_argument("--all-db", action="store_true", help="Validate all templates in database")
    parser.add_argument("--test-compatibility", action="store_true", help="Test hardware compatibility")
    parser.add_argument("--generator-type", type=str, choices=['merged', 'integrated_skillset', 'fixed', 'simple'],
                      help="Check compatibility with specific generator type")
    parser.add_argument("--output", type=str, help="Output file for validation results (JSON)")
    parser.add_argument("--report", type=str, help="Generate markdown report file")
    parser.add_argument("--store-in-db", action="store_true", help="Store validation results in database")
    parser.add_argument("--history", type=str, help="Show validation history for a model ID")
    parser.add_argument("--history-limit", type=int, default=10, help="Limit for validation history")
    parser.add_argument("--db-path", type=str, help="Path to template database")
    parser.add_argument("--validate-all-generators", action="store_true", 
                      help="Validate templates with all generator types")
    parser.add_argument("--verbose", action="store_true", help="Show verbose output")
    
    args = parser.parse_args()
    
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Track all results for possible report generation
    all_results = {}
    
    if args.file:
        # Validate a single file
        result = validate_template_file(args.file)
        all_results[os.path.basename(args.file)] = result
        
        if result['valid']:
            print(f"✅ Template {args.file} is valid.")
            print(f"   Supported platforms: {', '.join(result.get('supported_platforms', []))}")
        else:
            print(f"❌ Template {args.file} has {len(result['errors'])} errors:")
            for error in result['errors']:
                print(f"  - {error}")
    
    elif args.dir:
        # Validate all templates in a directory
        results = validate_template_directory(args.dir)
        all_results.update(results)
        
        # Count valid and invalid templates
        valid_count = sum(1 for r in results.values() if r['valid'])
        invalid_count = len(results) - valid_count
        
        print(f"\nValidation Results: {valid_count} valid, {invalid_count} invalid templates")
        
        # Show details for invalid templates
        if invalid_count > 0:
            print("\nInvalid Templates:")
            for template_name, result in results.items():
                if not result['valid']:
                    print(f"  - {template_name}:")
                    for error in result['errors'][:5]:  # Show first 5 errors
                        print(f"    - {error}")
                    if len(result['errors']) > 5:
                        print(f"    - ... and {len(result['errors']) - 5} more errors")
    
    elif args.model_id:
        # Validate a specific model from database
        result = validate_template_from_db(args.model_id, args.db_path)
        all_results[args.model_id] = result
        
        if result['valid']:
            print(f"✅ Template for model {args.model_id} is valid.")
            print(f"   Model name: {result.get('model_name')}")
            print(f"   Family: {result.get('model_family')}")
            print(f"   Modality: {result.get('modality')}")
            print(f"   Supported platforms: {', '.join(result.get('supported_platforms', []))}")
        else:
            print(f"❌ Template for model {args.model_id} has {len(result['errors'])} errors:")
            for error in result['errors']:
                print(f"  - {error}")
    
    elif args.all_db or args.family or args.modality:
        # Validate all templates in database, optionally filtered
        if args.validate_all_generators:
            # Run validation with all generator types
            generator_types = ['merged', 'integrated_skillset', 'fixed', 'simple']
            combined_results = {}
            
            for generator_type in generator_types:
                print(f"\nValidating with {generator_type} generator...")
                results = validate_all_db_templates(args.db_path, args.family, args.modality, generator_type)
                
                if isinstance(results, dict) and "error" in results:
                    print(f"Error: {results['error']}")
                    continue
                
                # Add generator type to result keys to avoid overwriting
                results_with_generator = {f"{k}_{generator_type}": v for k, v in results.items()}
                combined_results.update(results_with_generator)
            
            all_results.update(combined_results)
            
            # Count overall valid and invalid templates
            valid_count = sum(1 for r in combined_results.values() if r.get('valid', False))
            invalid_count = len(combined_results) - valid_count
            
            print(f"\nOverall Validation Results: {valid_count} valid, {invalid_count} invalid templates across all generators")
            
            # Show breakdown by generator type
            for generator_type in generator_types:
                generator_results = {k: v for k, v in combined_results.items() if k.endswith(f"_{generator_type}")}
                valid_gen = sum(1 for r in generator_results.values() if r.get('valid', False))
                invalid_gen = len(generator_results) - valid_gen
                
                if generator_results:
                    print(f"  - {generator_type} generator: {valid_gen} valid, {invalid_gen} invalid")
            
        else:
            # Validate with single generator type if specified
            results = validate_all_db_templates(args.db_path, args.family, args.modality, args.generator_type)
            
            if isinstance(results, dict) and "error" in results:
                print(f"Error: {results['error']}")
            else:
                all_results.update(results)
                
                # Count valid and invalid templates
                valid_count = sum(1 for r in results.values() if r.get('valid', False))
                invalid_count = len(results) - valid_count
                
                # Filter description
                filter_desc = []
                if args.family:
                    filter_desc.append(f"family='{args.family}'")
                if args.modality:
                    filter_desc.append(f"modality='{args.modality}'")
                if args.generator_type:
                    filter_desc.append(f"generator='{args.generator_type}'")
                
                filter_text = f" with {' and '.join(filter_desc)}" if filter_desc else ""
                
                print(f"\nValidation Results{filter_text}: {valid_count} valid, {invalid_count} invalid templates")
                
                # Show stats on platform support
                platform_counts = {
                    'cuda': 0,
                    'cpu': 0,
                    'mps': 0,
                    'rocm': 0,
                    'openvino': 0,
                    'qualcomm': 0,
                    'webnn': 0,
                    'webgpu': 0
                }
                
                for result in results.values():
                    for platform in result.get('supported_platforms', []):
                        if platform in platform_counts:
                            platform_counts[platform] += 1
                
                print("\nHardware Platform Support:")
                for platform, count in platform_counts.items():
                    percentage = count/len(results)*100 if results else 0
                    print(f"  - {platform}: {count} templates ({percentage:.1f}%)")
                
                # Show details for invalid templates
                if invalid_count > 0:
                    print("\nInvalid Templates:")
                    for model_id, result in results.items():
                        if not result.get('valid', True):
                            print(f"  - {model_id} ({result.get('model_family', 'unknown')}, {result.get('modality', 'unknown')}):")
                            for error in result.get('errors', [])[:3]:  # Show first 3 errors
                                print(f"    - {error}")
                            if len(result.get('errors', [])) > 3:
                                print(f"    - ... and {len(result.get('errors', [])) - 3} more errors")
    
    elif args.history:
        # Show validation history for a model
        history = get_validation_history(args.history, args.history_limit, args.db_path)
        
        if not history:
            print(f"No validation history found for model: {args.history}")
        elif isinstance(history, list) and "error" in history[0]:
            print(f"Error: {history[0]['error']}")
        else:
            print(f"\nValidation History for {args.history}:")
            for i, entry in enumerate(history):
                status = "✅ Valid" if entry.get('valid', False) else f"❌ Invalid ({entry.get('error_count', 0)} errors)"
                platforms = entry.get('supported_platforms', [])
                timestamp = entry.get('timestamp')
                print(f"{i+1}. [{timestamp}] {status} - Platforms: {', '.join(platforms)}")
    
    if args.test_compatibility and args.dir:
        # Test compatibility with available hardware
        compat_results = test_compatibility(args.dir)
        
        if isinstance(compat_results, dict) and "error" in compat_results:
            print(f"\nCompatibility Test Error: {compat_results['error']}")
        else:
            compatible_count = sum(1 for r in compat_results.values() if r.get('compatible', False))
            incompatible_count = len(compat_results) - compatible_count
            
            print(f"\nCompatibility Results: {compatible_count} compatible, {incompatible_count} incompatible")
            
            if incompatible_count > 0:
                print("\nIncompatible Templates:")
                for template_name, result in compat_results.items():
                    if not result.get('compatible', True):
                        print(f"  - {template_name}:")
                        for hw_type, details in result.get('details', {}).items():
                            if not details.get('compatible', True):
                                print(f"    - Incompatible with {hw_type}")
    
    # Generate report if requested
    if args.report and all_results:
        report = generate_validation_report(all_results)
        
        try:
            with open(args.report, 'w') as f:
                f.write(report)
            print(f"Validation report saved to: {args.report}")
        except Exception as e:
            print(f"Error writing report: {e}")
    
    # Store results if requested
    if args.output or args.store_in_db:
        if all_results:
            store_validation_results(all_results, args.output, args.store_in_db, args.db_path)
        else:
            print("No results to store")

if __name__ == "__main__":
    main()