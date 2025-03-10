#!/usr/bin/env python
# Template verification system for IPFS Accelerate
# Validates templates for syntax correctness and compatibility

import os
import sys
import logging
import importlib
import tempfile
import argparse
import ast
import re
from typing import Dict, List, Any, Tuple, Optional, Set
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Try importing template inheritance system
try:
    from template_inheritance_system import (
        register_template, register_template_directory,
        validate_all_templates, test_template_compatibility,
        get_template_inheritance_graph, TemplateError
    )
    template_system_available = True
except ImportError:
    logger.warning("Template inheritance system not available, using basic verification")
    template_system_available = False

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

def validate_template_file(file_path: str) -> Dict[str, Any]:
    """
    Validate a template file with multiple validation rules
    
    Args:
        file_path: Path to the template file
        
    Returns:
        Dictionary with validation results
    """
    logger.info(f"Validating template: {file_path}")
    
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
    
    # Run all validators
    validators = [
        ('syntax', validate_python_syntax),
        ('imports', validate_imports),
        ('class_structure', validate_class_structure),
        ('resource_pool', validate_resource_pool_usage),
        ('hardware_awareness', validate_hardware_awareness)
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
    
    # If template inheritance system is available, use it for additional validation
    template_inheritance_results = None
    if template_system_available:
        try:
            with tempfile.NamedTemporaryFile(suffix='.py', delete=False) as temp_file:
                temp_path = temp_file.name
                temp_file.write(content.encode('utf-8'))
            
            # Register and validate with template system
            try:
                template = register_template(temp_path, os.path.basename(file_path))
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
        'file': file_path,
        'errors': all_errors,
        'validators': results_by_validator
    }
    
    if template_inheritance_results:
        result['template_inheritance'] = template_inheritance_results
    
    return result

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
        if not (file_path.name.startswith('hf_') and file_path.name.endswith('template.py')):
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

def main():
    """Main function for standalone usage"""
    parser = argparse.ArgumentParser(description="Template verification system")
    parser.add_argument("--file", type=str, help="Validate a single template file")
    parser.add_argument("--dir", type=str, help="Validate all templates in a directory")
    parser.add_argument("--test-compatibility", action="store_true", help="Test hardware compatibility")
    parser.add_argument("--verbose", action="store_true", help="Show verbose output")
    
    args = parser.parse_args()
    
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    if args.file:
        # Validate a single file
        result = validate_template_file(args.file)
        if result['valid']:
            print(f"✅ Template {args.file} is valid.")
        else:
            print(f"❌ Template {args.file} has {len(result['errors'])} errors:")
            for error in result['errors']:
                print(f"  - {error}")
    
    elif args.dir:
        # Validate all templates in a directory
        results = validate_template_directory(args.dir)
        
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
                    for error in result['errors']:
                        print(f"    - {error}")
    
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

if __name__ == "__main__":
    main()