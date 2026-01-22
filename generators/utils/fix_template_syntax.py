#!/usr/bin/env python3
"""
Utility for fixing common syntax errors in templates.

This script analyzes and fixes common syntax errors in template files, 
such as mismatched brackets, indentation issues, and missing imports.

Usage:
    python fix_template_syntax.py --file TEMPLATE_FILE
    python fix_template_syntax.py --dir TEMPLATE_DIR
    python fix_template_syntax.py --db-path DB_PATH
"""

import os
import sys
import re
import argparse
import ast
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Check for DuckDB availability
try:
    import duckdb
    HAS_DUCKDB = True
except ImportError:
    HAS_DUCKDB = False
    logger.warning("DuckDB not available. Will use JSON-based storage.")

def verify_syntax(content: str) -> Tuple[bool, str]:
    """
    Check if Python code has syntax errors
    
    Args:
        content: Python code as string
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        ast.parse(content)
        return True, ""
    except SyntaxError as e:
        return False, f"Syntax error at line {e.lineno}: {e.msg}"

def fix_indentation(content: str) -> str:
    """
    Fix common indentation issues in code
    
    Args:
        content: Python code as string
        
    Returns:
        Fixed Python code
    """
    lines = content.split('\n')
    fixed_lines = []
    current_indent = 0
    in_class = False
    in_function = False
    expected_indent = 0
    
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped.startswith('#'):
            # Keep empty lines and comments as is
            fixed_lines.append(line)
            continue
        
        # Check for class or function definition
        if re.match(r'^\s*class\s+\w+', line):
            in_class = True
            current_indent = len(line) - len(line.lstrip())
            expected_indent = current_indent + 4  # Standard 4-space indent
        elif re.match(r'^\s*def\s+\w+', line):
            in_function = True
            current_indent = len(line) - len(line.lstrip())
            expected_indent = current_indent + 4  # Standard 4-space indent
        
        # Check if we're in a block that expects indented lines
        if in_class or in_function:
            if i > 0 and lines[i-1].endswith(':') and not line.endswith(':'):
                # Line after a colon should be indented
                leading_spaces = len(line) - len(line.lstrip())
                if leading_spaces < expected_indent:
                    # Fix indentation
                    line = ' ' * expected_indent + line.lstrip()
        
        # Check if we're exiting a block
        if in_class and re.match(r'^\s*class\s+\w+', line):
            in_class = False
        elif in_function and re.match(r'^\s*def\s+\w+', line):
            in_function = False
            
        fixed_lines.append(line)
    
    return '\n'.join(fixed_lines)

def fix_bracket_mismatch(content: str) -> str:
    """
    Fix mismatched brackets, parentheses, and braces
    
    Args:
        content: Python code as string
        
    Returns:
        Fixed Python code
    """
    lines = content.split('\n')
    # Look for mismatched brackets in dictionary definitions {}
    for i, line in enumerate(lines):
        # Simple case: check if there's an opening { without a closing }
        if '{' in line and '}' not in line:
            # Check for patterns like "variable = {" or "return {"
            if re.search(r'=\s*\{\s*$', line) or re.search(r'return\s+\{\s*$', line):
                # Look ahead for the closing bracket
                brace_count = 1
                for j in range(i+1, len(lines)):
                    brace_count += lines[j].count('{')
                    brace_count -= lines[j].count('}')
                    
                    if brace_count == 0:
                        break
                else:
                    # If we didn't find a matching closing brace,
                    # add one at the appropriate indentation
                    next_non_empty = i + 1
                    while next_non_empty < len(lines) and not lines[next_non_empty].strip():
                        next_non_empty += 1
                    
                    if next_non_empty < len(lines):
                        # Use the indentation of the original line
                        indent = len(line) - len(line.lstrip())
                        # Find where to insert the closing brace based on indentation
                        for j in range(next_non_empty, len(lines)):
                            if (len(lines[j]) - len(lines[j].lstrip())) <= indent:
                                lines.insert(j, ' ' * indent + '}')
                                break
                        else:
                            # If we didn't find a good spot, add at the end
                            lines.append(' ' * indent + '}')
    
    # Join the lines back together
    return '\n'.join(lines)

def add_missing_imports(content: str) -> str:
    """
    Add common missing imports to code
    
    Args:
        content: Python code as string
        
    Returns:
        Fixed Python code with added imports
    """
    required_imports = {
        'os': 'import os',
        'sys': 'import sys',
        'logging': 'import logging',
        'torch': 'import torch'
    }
    
    # Check existing imports
    imports = {}
    import_section = []
    non_import_line_found = False
    
    lines = content.split('\n')
    for i, line in enumerate(lines):
        if line.strip().startswith(('import ', 'from ')):
            import_section.append(i)
            if line.startswith('import '):
                module = line.split('import ')[1].split()[0].split('.')[0]
                imports[module] = line
            elif line.startswith('from '):
                module = line.split('from ')[1].split('import')[0].strip().split('.')[0]
                imports[module] = line
        elif line.strip() and not line.strip().startswith('#') and not line.strip().startswith('"""'):
            if import_section and not non_import_line_found:
                # First non-import line found after imports
                non_import_line_found = True
                import_insertion_point = i
    
    # If no imports found, insert at top (after any module docstring)
    if not import_section:
        # Check for module docstring
        has_docstring = False
        docstring_end = 0
        if len(lines) > 0 and lines[0].strip().startswith('"""'):
            has_docstring = True
            for i, line in enumerate(lines):
                if i > 0 and '"""' in line:
                    docstring_end = i + 1
                    break
            if docstring_end == 0:  # Docstring not closed
                docstring_end = 1  # Just insert after the first line
        
        import_insertion_point = docstring_end
    
    # Add missing imports
    added_imports = []
    for module, import_line in required_imports.items():
        if module not in imports:
            added_imports.append(import_line)
    
    if added_imports:
        if import_insertion_point == 0:
            # Add a blank line after imports if needed
            added_imports.append('')
            lines = added_imports + lines
        else:
            # Add imports at the insertion point
            lines = lines[:import_insertion_point] + added_imports + lines[import_insertion_point:]
    
    return '\n'.join(lines)

def fix_template_variables(content: str) -> str:
    """
    Fix common issues with template variables
    
    Args:
        content: Python code as string
        
    Returns:
        Fixed Python code with corrected template variables
    """
    # Ensure model_name variable is present
    if '{{model_name}}' not in content and '"model_name"' in content:
        content = content.replace('"model_name"', '"{{model_name}}"')
    
    # Fix malformed template variables
    content = re.sub(r'{\s*{([^}]+)}\s*}', r'{{{\1}}}', content)
    
    return content

def add_hardware_support(content: str) -> str:
    """
    Add hardware platform support to templates lacking it
    
    Args:
        content: Python code as string
        
    Returns:
        Fixed Python code with hardware support
    """
    # Check if template has hardware detection
    if ('cuda' in content and 'cpu' in content) or 'detect_available_hardware' in content:
        # Already has hardware support
        return content
    
    # Parse the AST to find class definitions
    try:
        tree = ast.parse(content)
        classes = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
        
        if not classes:
            # No classes to add hardware support to
            return content
        
        # Add hardware detection function to each class
        lines = content.split('\n')
        for cls in classes:
            # Check if class already has a setup_hardware method
            has_setup = any(
                isinstance(node, ast.FunctionDef) and node.name == 'setup_hardware'
                for node in cls.body
            )
            
            if not has_setup:
                # Find the end of the class to insert the setup_hardware method
                class_start_line = cls.lineno - 1  # AST line numbers are 1-based, list indices are 0-based
                
                # Find the indentation of the class body
                indent = 0
                for i in range(class_start_line + 1, len(lines)):
                    line = lines[i]
                    if line.strip():
                        indent = len(line) - len(line.lstrip())
                        break
                
                # Create the setup_hardware method
                setup_hardware_method = [
                    f"{' ' * indent}def setup_hardware(self):",
                    f"{' ' * (indent + 4)}\"\"\"Set up hardware detection for the template.\"\"\"",
                    f"{' ' * (indent + 4)}# CUDA support",
                    f"{' ' * (indent + 4)}self.has_cuda = torch.cuda.is_available()",
                    f"{' ' * (indent + 4)}# MPS support (Apple Silicon)",
                    f"{' ' * (indent + 4)}self.has_mps = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()",
                    f"{' ' * (indent + 4)}# ROCm support (AMD)",
                    f"{' ' * (indent + 4)}self.has_rocm = hasattr(torch, 'version') and hasattr(torch.version, 'hip') and torch.version.hip is not None",
                    f"{' ' * (indent + 4)}# OpenVINO support",
                    f"{' ' * (indent + 4)}self.has_openvino = 'openvino' in sys.modules",
                    f"{' ' * (indent + 4)}# Qualcomm AI Engine support",
                    f"{' ' * (indent + 4)}self.has_qualcomm = 'qti' in sys.modules or 'qnn_wrapper' in sys.modules",
                    f"{' ' * (indent + 4)}# WebNN/WebGPU support",
                    f"{' ' * (indent + 4)}self.has_webnn = False  # Will be set by WebNN bridge if available",
                    f"{' ' * (indent + 4)}self.has_webgpu = False  # Will be set by WebGPU bridge if available",
                    f"{' ' * (indent + 4)}",
                    f"{' ' * (indent + 4)}# Set default device",
                    f"{' ' * (indent + 4)}if self.has_cuda:",
                    f"{' ' * (indent + 4)}    self.device = 'cuda'",
                    f"{' ' * (indent + 4)}elif self.has_mps:",
                    f"{' ' * (indent + 4)}    self.device = 'mps'",
                    f"{' ' * (indent + 4)}elif self.has_rocm:",
                    f"{' ' * (indent + 4)}    self.device = 'cuda'  # ROCm uses CUDA compatibility layer",
                    f"{' ' * (indent + 4)}else:",
                    f"{' ' * (indent + 4)}    self.device = 'cpu'"
                ]
                
                # Find the __init__ method to call setup_hardware
                has_init = False
                for init_node in [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == '__init__']:
                    has_init = True
                    # Find the init method's body to add setup_hardware call
                    init_start_line = init_node.lineno - 1
                    init_end_line = init_node.end_lineno if hasattr(init_node, 'end_lineno') else -1
                    
                    if init_end_line != -1:
                        # Check if setup_hardware is already called
                        init_body = '\n'.join(lines[init_start_line:init_end_line])
                        if 'setup_hardware' not in init_body:
                            # Find a good place to insert the call (before the end of the method)
                            for i in range(init_end_line - 1, init_start_line, -1):
                                if lines[i].strip():
                                    # Insert after the last non-empty line
                                    lines.insert(i + 1, f"{' ' * (indent + 4)}self.setup_hardware()")
                                    break
                
                # If no __init__ method, add one
                if not has_init:
                    init_method = [
                        f"{' ' * indent}def __init__(self):",
                        f"{' ' * (indent + 4)}\"\"\"Initialize the template.\"\"\"",
                        f"{' ' * (indent + 4)}self.model_name = \"{{{{model_name}}}}\"",
                        f"{' ' * (indent + 4)}self.setup_hardware()"
                    ]
                    
                    # Find a good place to insert the init method (beginning of class)
                    for i in range(class_start_line + 1, len(lines)):
                        line = lines[i].strip()
                        if line and not line.startswith('#') and not line.startswith('"""'):
                            # Insert before the first non-comment, non-docstring line
                            for j, init_line in enumerate(init_method):
                                lines.insert(i + j, init_line)
                            break
                
                # Insert the setup_hardware method at the end of the class
                # Find the end of the class
                class_indent = len(lines[class_start_line]) - len(lines[class_start_line].lstrip())
                class_end_line = len(lines)
                for i in range(class_start_line + 1, len(lines)):
                    if lines[i].strip() and (len(lines[i]) - len(lines[i].lstrip())) <= class_indent:
                        # Found a line with same or less indentation than class
                        class_end_line = i
                        break
                
                # Insert before the end of the class
                for i, setup_line in enumerate(setup_hardware_method):
                    lines.insert(class_end_line + i, setup_line)
        
        return '\n'.join(lines)
    except SyntaxError:
        # If there are syntax errors, we can't parse the AST
        # Return the original content
        return content

def fix_template_file(file_path: str) -> Tuple[bool, str]:
    """
    Fix a template file
    
    Args:
        file_path: Path to the template file
        
    Returns:
        Tuple of (success, error_message)
    """
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Verify original syntax
        valid, error = verify_syntax(content)
        if valid:
            logger.info(f"Template {file_path} has valid syntax, checking for improvements")
        else:
            logger.warning(f"Template {file_path} has syntax errors: {error}")
        
        # Apply fixes
        fixed_content = content
        fixed_content = fix_indentation(fixed_content)
        fixed_content = fix_bracket_mismatch(fixed_content)
        fixed_content = add_missing_imports(fixed_content)
        fixed_content = fix_template_variables(fixed_content)
        fixed_content = add_hardware_support(fixed_content)
        
        # Verify fixed syntax
        valid, error = verify_syntax(fixed_content)
        if valid:
            logger.info(f"Fixed template {file_path} now has valid syntax")
            
            # Save the fixed template
            backup_path = f"{file_path}.bak"
            with open(backup_path, 'w') as f:
                f.write(content)
            logger.info(f"Saved backup to {backup_path}")
            
            with open(file_path, 'w') as f:
                f.write(fixed_content)
            logger.info(f"Saved fixed template to {file_path}")
            
            return True, "Fixed successfully"
        else:
            logger.error(f"Failed to fix template {file_path}: {error}")
            return False, f"Failed to fix: {error}"
    
    except Exception as e:
        logger.error(f"Error fixing template {file_path}: {str(e)}")
        return False, str(e)

def fix_templates_in_directory(directory_path: str) -> Dict[str, Any]:
    """
    Fix all template files in a directory
    
    Args:
        directory_path: Directory containing template files
        
    Returns:
        Dictionary with results
    """
    results = {
        'success': True,
        'total': 0,
        'fixed': 0,
        'failed': 0,
        'details': {}
    }
    
    try:
        if not os.path.exists(directory_path):
            results['success'] = False
            results['error'] = f"Directory not found: {directory_path}"
            return results
        
        # Find template files
        template_files = []
        for file_path in Path(directory_path).glob('**/*.py'):
            # Skip files starting with underscore
            if file_path.name.startswith('_'):
                continue
            
            # Only include template files
            if file_path.name.startswith('template_') or 'template' in file_path.name.lower():
                template_files.append(str(file_path))
        
        logger.info(f"Found {len(template_files)} template files in {directory_path}")
        results['total'] = len(template_files)
        
        # Fix each template file
        for file_path in template_files:
            logger.info(f"Fixing template: {file_path}")
            success, message = fix_template_file(file_path)
            
            results['details'][file_path] = {
                'success': success,
                'message': message
            }
            
            if success:
                results['fixed'] += 1
            else:
                results['failed'] += 1
                results['success'] = False
        
        return results
    
    except Exception as e:
        logger.error(f"Error fixing templates in directory {directory_path}: {str(e)}")
        results['success'] = False
        results['error'] = str(e)
        return results

def fix_templates_in_db(db_path: str) -> Dict[str, Any]:
    """
    Fix all templates in a database
    
    Args:
        db_path: Path to the database file
        
    Returns:
        Dictionary with results
    """
    results = {
        'success': True,
        'total': 0,
        'fixed': 0,
        'failed': 0,
        'details': {}
    }
    
    # Check if using JSON-based storage
    if not HAS_DUCKDB or db_path.endswith('.json'):
        json_db_path = db_path if db_path.endswith('.json') else db_path.replace('.duckdb', '.json')
        
        try:
            if not os.path.exists(json_db_path):
                results['success'] = False
                results['error'] = f"JSON database file not found: {json_db_path}"
                return results
            
            # Load the JSON database
            with open(json_db_path, 'r') as f:
                template_db = json.load(f)
            
            if 'templates' not in template_db:
                results['success'] = False
                results['error'] = "No templates found in JSON database"
                return results
            
            templates = template_db['templates']
            if not templates:
                results['success'] = False
                results['error'] = "No templates found in JSON database"
                return results
            
            logger.info(f"Found {len(templates)} templates in JSON database")
            results['total'] = len(templates)
            
            # Fix each template
            for template_id, template_data in templates.items():
                content = template_data.get('template', '')
                model_type = template_data.get('model_type', 'unknown')
                template_type = template_data.get('template_type', 'unknown')
                platform = template_data.get('platform')
                
                platform_str = f"{model_type}/{template_type}"
                if platform:
                    platform_str += f"/{platform}"
                
                logger.info(f"Fixing template {template_id}: {platform_str}")
                
                # Verify original syntax
                valid, error = verify_syntax(content)
                if valid:
                    logger.info(f"Template {template_id} has valid syntax, checking for improvements")
                else:
                    logger.warning(f"Template {template_id} has syntax errors: {error}")
                
                # Apply fixes
                fixed_content = content
                fixed_content = fix_indentation(fixed_content)
                fixed_content = fix_bracket_mismatch(fixed_content)
                fixed_content = add_missing_imports(fixed_content)
                fixed_content = fix_template_variables(fixed_content)
                fixed_content = add_hardware_support(fixed_content)
                
                # Verify fixed syntax
                valid, error = verify_syntax(fixed_content)
                if valid:
                    logger.info(f"Fixed template {template_id} now has valid syntax")
                    
                    # Update the template in the database
                    template_db['templates'][template_id]['template'] = fixed_content
                    template_db['templates'][template_id]['updated_at'] = datetime.now().isoformat()
                    
                    results['details'][template_id] = {
                        'success': True,
                        'message': "Fixed successfully"
                    }
                    results['fixed'] += 1
                else:
                    logger.error(f"Failed to fix template {template_id}: {error}")
                    results['details'][template_id] = {
                        'success': False,
                        'message': f"Failed to fix: {error}"
                    }
                    results['failed'] += 1
                    results['success'] = False
            
            # Save the updated database
            with open(json_db_path, 'w') as f:
                json.dump(template_db, f, indent=2)
            
            logger.info(f"Saved fixed templates to {json_db_path}")
            return results
        
        except Exception as e:
            logger.error(f"Error fixing templates in JSON database {json_db_path}: {str(e)}")
            results['success'] = False
            results['error'] = str(e)
            return results
    else:
        # Use DuckDB
        try:
            if not os.path.exists(db_path):
                results['success'] = False
                results['error'] = f"Database file not found: {db_path}"
                return results
            
            # Connect to the database
            conn = duckdb.connect(db_path)
            
            # Check if templates table exists
            table_check = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='templates'").fetchall()
            if not table_check:
                results['success'] = False
                results['error'] = "No 'templates' table found in database"
                return results
            
            # Get all templates
            templates = conn.execute("SELECT id, model_type, template_type, platform, template FROM templates").fetchall()
            if not templates:
                results['success'] = False
                results['error'] = "No templates found in database"
                return results
            
            logger.info(f"Found {len(templates)} templates in database")
            results['total'] = len(templates)
            
            # Fix each template
            for template in templates:
                template_id, model_type, template_type, platform, content = template
                
                platform_str = f"{model_type}/{template_type}"
                if platform:
                    platform_str += f"/{platform}"
                
                logger.info(f"Fixing template {template_id}: {platform_str}")
                
                # Verify original syntax
                valid, error = verify_syntax(content)
                if valid:
                    logger.info(f"Template {template_id} has valid syntax, checking for improvements")
                else:
                    logger.warning(f"Template {template_id} has syntax errors: {error}")
                
                # Apply fixes
                fixed_content = content
                fixed_content = fix_indentation(fixed_content)
                fixed_content = fix_bracket_mismatch(fixed_content)
                fixed_content = add_missing_imports(fixed_content)
                fixed_content = fix_template_variables(fixed_content)
                fixed_content = add_hardware_support(fixed_content)
                
                # Verify fixed syntax
                valid, error = verify_syntax(fixed_content)
                if valid:
                    logger.info(f"Fixed template {template_id} now has valid syntax")
                    
                    # Update the template in the database
                    conn.execute("""
                    UPDATE templates
                    SET template = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                    """, [fixed_content, template_id])
                    
                    results['details'][str(template_id)] = {
                        'success': True,
                        'message': "Fixed successfully"
                    }
                    results['fixed'] += 1
                else:
                    logger.error(f"Failed to fix template {template_id}: {error}")
                    results['details'][str(template_id)] = {
                        'success': False,
                        'message': f"Failed to fix: {error}"
                    }
                    results['failed'] += 1
                    results['success'] = False
            
            # Commit changes and close connection
            conn.close()
            
            logger.info(f"Saved fixed templates to {db_path}")
            return results
        
        except Exception as e:
            logger.error(f"Error fixing templates in database {db_path}: {str(e)}")
            results['success'] = False
            results['error'] = str(e)
            return results

def main():
    """Main function for standalone usage"""
    parser = argparse.ArgumentParser(description="Template Syntax Fixer")
    parser.add_argument("--file", type=str, help="Fix a single template file")
    parser.add_argument("--dir", type=str, help="Fix all templates in a directory")
    parser.add_argument("--db-path", type=str, help="Fix all templates in a database")
    
    args = parser.parse_args()
    
    if args.file:
        success, message = fix_template_file(args.file)
        if success:
            print(f"✅ Successfully fixed template: {args.file}")
        else:
            print(f"❌ Failed to fix template: {args.file}")
            print(f"   Error: {message}")
    
    elif args.dir:
        results = fix_templates_in_directory(args.dir)
        print(f"\nTemplate Directory Fix Results:")
        print(f"  - Total templates: {results['total']}")
        print(f"  - Successfully fixed: {results['fixed']}")
        print(f"  - Failed to fix: {results['failed']}")
        
        if results['failed'] > 0:
            print("\nFailed templates:")
            for file_path, details in results['details'].items():
                if not details['success']:
                    print(f"  - {file_path}: {details['message']}")
    
    elif args.db_path:
        results = fix_templates_in_db(args.db_path)
        print(f"\nTemplate Database Fix Results:")
        print(f"  - Total templates: {results['total']}")
        print(f"  - Successfully fixed: {results['fixed']}")
        print(f"  - Failed to fix: {results['failed']}")
        
        if results['failed'] > 0:
            print("\nFailed templates:")
            for template_id, details in results['details'].items():
                if not details['success']:
                    print(f"  - {template_id}: {details['message']}")
    
    else:
        parser.print_help()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())