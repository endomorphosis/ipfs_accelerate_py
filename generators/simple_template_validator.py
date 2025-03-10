#!/usr/bin/env python
"""
Simple template validator for hardware test templates.

This validator checks templates for correct syntax, imports, class structure,
hardware awareness, and template variables. It can validate individual files,
directories, or templates stored in a DuckDB database.
"""

# Check for DuckDB availability
import importlib.util
HAS_DUCKDB = importlib.util.find_spec("duckdb") is not None

import os
import sys
import argparse
import re
import ast
from typing import Dict, List, Any, Tuple
from pathlib import Path

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
        
        # Check for methods in test classes
        for cls in test_classes:
            methods = [node.name for node in cls.body if isinstance(node, ast.FunctionDef)]
            
            # Check for at least one test method
            test_methods = [m for m in methods if m.startswith('test_') or m == 'run']
            if not test_methods:
                errors.append(f"Class {cls.name} has no test methods (should start with 'test_' or include 'run' method)")
        
        return len(errors) == 0, errors
    except SyntaxError:
        # This should be caught by validate_python_syntax
        errors.append("Cannot analyze class structure due to syntax errors")
        return False, errors

def validate_hardware_awareness(content: str) -> Tuple[bool, List[str], List[str]]:
    """
    Validate hardware awareness and cross-platform support in the template
    
    Args:
        content: Template content as string
        
    Returns:
        Tuple of (valid, list of errors, list of supported platforms)
    """
    errors = []
    warnings = []
    
    # Define hardware platforms to check for
    hardware_platforms = [
        ('cuda', r'(cuda|gpu)'),
        ('cpu', r'cpu'),
        ('mps', r'(mps|apple|m1|m2)'),
        ('rocm', r'(rocm|amd)'),
        ('openvino', r'(openvino|intel)'),
        ('qualcomm', r'(qualcomm|qnn|hexagon)'),
        ('webnn', r'webnn'),
        ('webgpu', r'webgpu')
    ]
    
    supported_platforms = []
    
    # More sophisticated hardware detection checks
    hardware_checks = {
        'cuda': [
            r'HAS_CUDA', 
            r'torch\.cuda\.is_available', 
            r'device\s*=\s*[\'"]cuda[\'"]'
        ],
        'cpu': [
            r'device\s*=\s*[\'"]cpu[\'"]'
        ],
        'mps': [
            r'HAS_MPS', 
            r'torch\.mps\.is_available', 
            r'device\s*=\s*[\'"]mps[\'"]'
        ],
        'rocm': [
            r'HAS_ROCM', 
            r'_rocm_version', 
            r'ROCM_HOME'
        ],
        'openvino': [
            r'HAS_OPENVINO',
            r'openvino',
            r'import\s+openvino'
        ],
        'qualcomm': [
            r'HAS_QUALCOMM',
            r'QUALCOMM_SDK',
            r'qnn_wrapper',
            r'import\s+qti'
        ],
        'webnn': [
            r'HAS_WEBNN',
            r'WEBNN_AVAILABLE',
            r'WEBNN_SIMULATION'
        ],
        'webgpu': [
            r'HAS_WEBGPU',
            r'WEBGPU_AVAILABLE',
            r'WEBGPU_SIMULATION'
        ]
    }
    
    # Check for centralized hardware detection
    uses_central_detection = False
    if 'detect_available_hardware' in content or 'hardware_detection' in content:
        uses_central_detection = True
        print("Found centralized hardware detection")
    
    # Check for explicit hardware checks
    for platform, patterns in hardware_checks.items():
        for pattern in patterns:
            if re.search(pattern, content):
                if platform not in supported_platforms:
                    supported_platforms.append(platform)
                break
    
    # If still not found hardware, check for mentions
    if not supported_platforms:
        for platform_name, pattern in hardware_platforms:
            if re.search(pattern, content, re.IGNORECASE):
                supported_platforms.append(platform_name)
    
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
    
    # Check for Qualcomm support (new in March 2025)
    has_qualcomm = 'qualcomm' in supported_platforms
    if not has_qualcomm:
        warnings.append("No Qualcomm AI Engine support detected")
    
    # Check for all platforms
    if len(supported_platforms) < 3:
        warnings.append(f"Limited hardware support detected, only found: {', '.join(supported_platforms)}")
    
    # Add warnings to errors
    if warnings:
        errors.extend([f"WARNING: {w}" for w in warnings])
    
    # Only count serious errors (not warnings) for validity
    success = not any(not e.startswith("WARNING:") for e in errors)
    
    return success, errors, supported_platforms

def validate_template_file(file_path: str) -> Dict[str, Any]:
    """
    Validate a template file with multiple validation rules
    
    Args:
        file_path: Path to the template file
        
    Returns:
        Dictionary with validation results
    """
    print(f"Validating template: {file_path}")
    
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
    
    # Run hardware awareness check separately to capture supported platforms
    hw_valid, hw_errors, supported_platforms = validate_hardware_awareness(content)
    results_by_validator['hardware_awareness'] = {
        'valid': hw_valid,
        'errors': hw_errors,
        'supported_platforms': supported_platforms
    }
    all_valid = all_valid and hw_valid
    all_errors.extend([f"hardware_awareness: {error}" for error in hw_errors])
    
    # Combine all results
    result = {
        'valid': all_valid,
        'file': file_path,
        'errors': all_errors,
        'validators': results_by_validator,
        'supported_platforms': supported_platforms
    }
    
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
        if not (file_path.name.startswith('template_') or file_path.name.endswith('template.py')):
            continue
        
        # Validate the template
        result = validate_template_file(str(file_path))
        results[file_path.name] = result
    
    return results

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
    required_vars = ['model_name']
    
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

def validate_duckdb_templates(db_path: str = "template_db.duckdb") -> Dict[str, Any]:
    """
    Validate templates stored in a DuckDB database
    
    Args:
        db_path: Path to the DuckDB database
        
    Returns:
        Dictionary with validation results
    """
    if not HAS_DUCKDB:
        return {
            'valid': False,
            'error': "DuckDB not available. Install with 'pip install duckdb'"
        }
    
    try:
        import duckdb
    except ImportError:
        return {
            'valid': False,
            'error': "Failed to import DuckDB"
        }
    
    if not os.path.exists(db_path):
        return {
            'valid': False,
            'error': f"Database file not found: {db_path}"
        }
    
    try:
        conn = duckdb.connect(db_path)
        
        # Check if templates table exists
        table_check = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='templates'").fetchall()
        if not table_check:
            return {
                'valid': False,
                'error': "No 'templates' table found in database"
            }
        
        # Get all templates
        templates = conn.execute("SELECT id, model_type, template_type, platform, template FROM templates").fetchall()
        if not templates:
            return {
                'valid': False,
                'error': "No templates found in database"
            }
        
        print(f"Found {len(templates)} templates in database")
        
        results = {}
        valid_count = 0
        
        # Validate each template
        for template in templates:
            template_id, model_type, template_type, platform, content = template
            
            platform_str = f"{model_type}/{template_type}"
            if platform:
                platform_str += f"/{platform}"
            
            print(f"Validating template {template_id}: {platform_str}")
            
            # Run validations
            validators = [
                ('syntax', validate_python_syntax),
                ('imports', validate_imports),
                ('class_structure', validate_class_structure),
                ('template_vars', validate_template_variables)
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
            
            # Run hardware awareness check separately
            hw_valid, hw_errors, supported_platforms = validate_hardware_awareness(content)
            results_by_validator['hardware_awareness'] = {
                'valid': hw_valid,
                'errors': hw_errors,
                'supported_platforms': supported_platforms
            }
            all_valid = all_valid and hw_valid
            all_errors.extend([f"hardware_awareness: {error}" for error in hw_errors])
            
            # Store results
            template_key = f"{template_id}_{model_type}_{template_type}"
            if platform:
                template_key += f"_{platform}"
                
            results[template_key] = {
                'id': template_id,
                'model_type': model_type,
                'template_type': template_type,
                'platform': platform,
                'valid': all_valid,
                'errors': all_errors,
                'validators': results_by_validator,
                'supported_platforms': supported_platforms
            }
            
            if all_valid:
                valid_count += 1
        
        conn.close()
        
        return {
            'valid': valid_count == len(templates),
            'total': len(templates),
            'valid_count': valid_count,
            'invalid_count': len(templates) - valid_count,
            'templates': results
        }
        
    except Exception as e:
        return {
            'valid': False,
            'error': f"Error validating database templates: {str(e)}"
        }

def main():
    """Main function for standalone usage"""
    parser = argparse.ArgumentParser(description="Simple Template Validator")
    parser.add_argument("--file", type=str, help="Validate a single template file")
    parser.add_argument("--dir", type=str, help="Validate all templates in a directory")
    parser.add_argument("--db", type=str, help="Validate templates in a DuckDB database (default: template_db.duckdb)", 
                      default="template_db.duckdb")
    parser.add_argument("--validate-db", action="store_true", help="Validate templates in the database")
    args = parser.parse_args()
    
    if args.file:
        # Validate a single file
        result = validate_template_file(args.file)
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
        
        # Count valid and invalid templates
        valid_count = sum(1 for r in results.values() if r['valid'])
        invalid_count = len(results) - valid_count
        
        print(f"\nValidation Results: {valid_count} valid, {invalid_count} invalid templates")
        
        # Platform support stats
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
            for template_name, result in results.items():
                if not result['valid']:
                    print(f"  - {template_name}:")
                    for error in result['errors'][:5]:  # Show first 5 errors
                        print(f"    - {error}")
                    if len(result['errors']) > 5:
                        print(f"    - ... and {len(result['errors']) - 5} more errors")
    
    elif args.validate_db:
        # Validate templates in the database
        if not HAS_DUCKDB:
            print("❌ Error: DuckDB not available. Install with 'pip install duckdb'")
            return 1
            
        print(f"Validating templates in database: {args.db}")
        db_results = validate_duckdb_templates(args.db)
        
        if 'error' in db_results:
            print(f"❌ Error: {db_results['error']}")
            return 1
            
        valid_count = db_results['valid_count']
        invalid_count = db_results['invalid_count']
        total = db_results['total']
        
        print(f"\nValidation Results: {valid_count} valid, {invalid_count} invalid templates (out of {total})")
        
        # Platform support stats
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
        
        for result in db_results['templates'].values():
            for platform in result.get('supported_platforms', []):
                if platform in platform_counts:
                    platform_counts[platform] += 1
        
        print("\nHardware Platform Support:")
        for platform, count in platform_counts.items():
            percentage = count/total*100 if total else 0
            print(f"  - {platform}: {count} templates ({percentage:.1f}%)")
            
        # Show details for invalid templates
        if invalid_count > 0:
            print("\nInvalid Templates:")
            invalid_templates = {k: v for k, v in db_results['templates'].items() if not v['valid']}
            
            for name, result in invalid_templates.items():
                model_type = result['model_type']
                template_type = result['template_type']
                platform = result['platform'] or 'all'
                template_id = result['id']
                
                print(f"  - Template {template_id} ({model_type}/{template_type}/{platform}):")
                for error in result['errors'][:5]:  # Show first 5 errors
                    print(f"    - {error}")
                if len(result['errors']) > 5:
                    print(f"    - ... and {len(result['errors']) - 5} more errors")
    else:
        parser.print_help()

if __name__ == "__main__":
    main()