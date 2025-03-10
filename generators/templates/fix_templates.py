#!/usr/bin/env python3
"""
Template syntax fixer script.

This script fixes syntax errors in templates that have been identified as problematic
in the NEXT_STEPS.md file. It uses Python's ast module to validate syntax and
provides common fixes for bracket mismatches, indentation issues, and other problems.
"""

import os
import sys
import re
import json
import logging
import argparse
import ast
import tempfile
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path

# Import the template database manager
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from template_database import TemplateDatabase

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Templates with known issues from NEXT_STEPS.md
PROBLEMATIC_TEMPLATES = [
    # Format: (model_type, template_type, hardware_platform)
    ("video", "test", None),
    ("cpu", "test", "cpu_embedding"),
    ("llama", "test", None),
    ("text_embedding", "test", None),
    ("t5", "test", None),
    ("xclip", "test", None),
    ("clip", "test", None),
    ("test", "test", "test_generator"),
    ("vision", "test", None),
    ("detr", "test", None),
    ("qwen2", "test", None),
    ("vit", "test", None)
]

def validate_template(template: str) -> Tuple[bool, Optional[str]]:
    """Validate a template and check for syntax errors.
    
    Args:
        template: Template string to validate
        
    Returns:
        (True, None) if valid, (False, error_message) if invalid
    """
    try:
        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix='.py', delete=False) as tmp:
            tmp.write(template.encode('utf-8'))
            tmp_path = tmp.name
        
        # Try to parse with ast to check for syntax errors
        with open(tmp_path, 'r') as f:
            ast.parse(f.read())
        
        # Clean up
        os.unlink(tmp_path)
        
        return (True, None)
    except Exception as e:
        # Clean up
        if 'tmp_path' in locals():
            try:
                os.unlink(tmp_path)
            except:
                pass
        
        return (False, str(e))

def fix_bracket_mismatch(template: str) -> str:
    """Fix mismatched brackets in a template.
    
    Common issues:
    - Missing closing brackets
    - Extra closing brackets
    - Curly braces for string formatting vs dict literals
    
    Args:
        template: Template string with potential bracket issues
        
    Returns:
        Fixed template string
    """
    # Keep track of brackets
    bracket_counts = {
        '(': 0,
        '[': 0,
        '{': 0
    }
    
    bracket_pairs = {
        '(': ')',
        '[': ']',
        '{': '}'
    }
    
    reverse_pairs = {
        ')': '(',
        ']': '[',
        '}': '{'
    }
    
    lines = template.split('\n')
    fixed_lines = []
    
    # Process each line
    for line in lines:
        # Skip comment lines
        if line.strip().startswith('#'):
            fixed_lines.append(line)
            continue
        
        # Track brackets in this line
        for char in line:
            if char in bracket_counts:
                bracket_counts[char] += 1
            elif char in reverse_pairs:
                if bracket_counts[reverse_pairs[char]] > 0:
                    bracket_counts[reverse_pairs[char]] -= 1
                else:
                    # Extra closing bracket - remove it
                    line = line.replace(char, '', 1)
        
        fixed_lines.append(line)
    
    # Add missing closing brackets at the end
    last_line = fixed_lines[-1] if fixed_lines else ""
    
    # Add closing brackets in reverse order
    for bracket, count in bracket_counts.items():
        if count > 0:
            if last_line.strip():
                last_line += bracket_pairs[bracket] * count
            else:
                last_line = bracket_pairs[bracket] * count
    
    if fixed_lines:
        fixed_lines[-1] = last_line
    else:
        fixed_lines.append(last_line)
    
    return '\n'.join(fixed_lines)

def fix_unexpected_indent(template: str) -> str:
    """Fix unexpected indentation issues in a template.
    
    Common issues:
    - Inconsistent indentation levels
    - Tabs vs spaces
    - Unexpected indentation after block statements
    
    Args:
        template: Template string with potential indentation issues
        
    Returns:
        Fixed template string
    """
    lines = template.split('\n')
    fixed_lines = []
    
    # Replace tabs with spaces
    for i, line in enumerate(lines):
        # Replace tabs with 4 spaces
        lines[i] = line.replace('\t', '    ')
    
    # Track expected indent level
    current_indent = 0
    indent_stack = [0]
    
    # Process each line
    for i, line in enumerate(lines):
        # Skip empty lines and comments
        if not line.strip() or line.strip().startswith('#'):
            fixed_lines.append(line)
            continue
        
        # Get line's indentation
        indent = len(line) - len(line.lstrip())
        content = line.strip()
        
        # Check for block start (if, for, def, class, try, with, else, elif, except, finally)
        if (content.startswith(('def ', 'class ', 'if ', 'for ', 'while ', 'try:', 'with ')) or
            content == 'else:' or content.startswith(('elif ', 'except')) or
            content == 'finally:'):
            
            # Check if we need to fix indentation
            if indent != current_indent:
                line = ' ' * current_indent + content
            
            # Increase expected indent for next line
            if content.endswith(':'):
                indent_stack.append(current_indent + 4)
                current_indent = current_indent + 4
        
        # Check for block end
        elif indent < current_indent and indent in indent_stack:
            # Pop back to a previous indent level
            while indent_stack and indent_stack[-1] > indent:
                indent_stack.pop()
            
            current_indent = indent_stack[-1]
        
        # Fix line indentation if needed
        if indent != current_indent and line.strip():
            line = ' ' * current_indent + line.strip()
        
        fixed_lines.append(line)
    
    return '\n'.join(fixed_lines)

def fix_missing_indented_block(template: str) -> str:
    """Fix missing indented block after a block statement.
    
    Common issues:
    - Missing indented block after if, for, while, def, etc.
    
    Args:
        template: Template string with potential missing block issues
        
    Returns:
        Fixed template string
    """
    lines = template.split('\n')
    fixed_lines = []
    
    # Process each line
    for i, line in enumerate(lines):
        fixed_lines.append(line)
        
        # Check for block statement without following indented block
        if (line.strip().endswith(':') and
            i < len(lines) - 1 and
            not lines[i+1].strip().startswith('#') and
            len(lines[i+1]) - len(lines[i+1].lstrip()) <= len(line) - len(line.lstrip())):
            
            # Add a pass statement with proper indentation
            current_indent = len(line) - len(line.lstrip())
            fixed_lines.append(' ' * (current_indent + 4) + 'pass')
    
    return '\n'.join(fixed_lines)

def fix_template(template: str) -> Tuple[str, bool, Optional[str]]:
    """Apply all fixes to a template and validate the result.
    
    Args:
        template: Template string with potential issues
        
    Returns:
        (fixed_template, is_valid, error_message)
    """
    # Apply fixes
    fixed = template
    fixed = fix_bracket_mismatch(fixed)
    fixed = fix_unexpected_indent(fixed)
    fixed = fix_missing_indented_block(fixed)
    
    # Validate the fixed template
    valid, error = validate_template(fixed)
    
    return fixed, valid, error

def fix_all_templates(db_path: str = None, json_path: str = None,
                     specific_templates: List[Tuple[str, str, Optional[str]]] = None) -> Dict[str, Any]:
    """Fix all templates in the database or a specific set of templates.
    
    Args:
        db_path: Path to the DuckDB database
        json_path: Path to the JSON template file
        specific_templates: List of (model_type, template_type, hardware_platform) to fix
        
    Returns:
        Dictionary with results
    """
    # Initialize template database
    db = TemplateDatabase(db_path, json_path)
    
    # Results tracking
    results = {
        'total': 0,
        'fixed': 0,
        'failed': 0,
        'skipped': 0,
        'details': []
    }
    
    # Get templates to fix
    templates_to_fix = specific_templates or PROBLEMATIC_TEMPLATES
    
    # Process each template
    for model_type, template_type, hardware in templates_to_fix:
        results['total'] += 1
        
        # Get the template
        template = db.get_template(model_type, template_type, hardware)
        
        if template is None:
            logger.warning(f"Template not found: {model_type}/{template_type}/{hardware}")
            results['skipped'] += 1
            results['details'].append({
                'model_type': model_type,
                'template_type': template_type,
                'hardware': hardware,
                'status': 'skipped',
                'reason': 'Template not found'
            })
            continue
        
        # Validate the template
        valid, error = validate_template(template)
        
        if valid:
            logger.info(f"Template already valid: {model_type}/{template_type}/{hardware}")
            results['skipped'] += 1
            results['details'].append({
                'model_type': model_type,
                'template_type': template_type,
                'hardware': hardware,
                'status': 'skipped',
                'reason': 'Already valid'
            })
            continue
        
        # Try to fix the template
        fixed_template, fixed_valid, fixed_error = fix_template(template)
        
        if fixed_valid:
            # Update the template in the database
            success = db.add_template(model_type, template_type, fixed_template, hardware)
            
            if success:
                logger.info(f"Fixed template: {model_type}/{template_type}/{hardware}")
                results['fixed'] += 1
                results['details'].append({
                    'model_type': model_type,
                    'template_type': template_type,
                    'hardware': hardware,
                    'status': 'fixed',
                    'original_error': error
                })
            else:
                logger.error(f"Failed to update template: {model_type}/{template_type}/{hardware}")
                results['failed'] += 1
                results['details'].append({
                    'model_type': model_type,
                    'template_type': template_type,
                    'hardware': hardware,
                    'status': 'failed',
                    'reason': 'Database update failed',
                    'original_error': error
                })
        else:
            logger.error(f"Failed to fix template: {model_type}/{template_type}/{hardware}")
            results['failed'] += 1
            results['details'].append({
                'model_type': model_type,
                'template_type': template_type,
                'hardware': hardware,
                'status': 'failed',
                'reason': 'Auto-fix failed',
                'original_error': error,
                'fixed_error': fixed_error
            })
    
    return results

def add_missing_hardware_support(db_path: str = None, json_path: str = None,
                               model_types: List[str] = None) -> Dict[str, Any]:
    """Add missing hardware platform support to templates.
    
    Args:
        db_path: Path to the DuckDB database
        json_path: Path to the JSON template file
        model_types: List of model types to update (default: all)
        
    Returns:
        Dictionary with results
    """
    # Initialize template database
    db = TemplateDatabase(db_path, json_path)
    
    # Get all templates
    templates = db.list_templates()
    
    # Filter by model type if specified
    if model_types:
        templates = [t for t in templates if t['model_type'] in model_types]
    
    # Group templates by model type and template type
    template_groups = {}
    for t in templates:
        key = (t['model_type'], t['template_type'])
        if key not in template_groups:
            template_groups[key] = []
        template_groups[key].append(t)
    
    # Results tracking
    results = {
        'total_groups': len(template_groups),
        'updated': 0,
        'skipped': 0,
        'failed': 0,
        'details': []
    }
    
    # Process each template group
    for (model_type, template_type), group in template_groups.items():
        # Skip default templates
        if model_type == 'default':
            continue
        
        # Get supported hardware platforms
        supported = set()
        for t in group:
            if t['hardware_platform'] != 'all':
                supported.add(t['hardware_platform'])
        
        # Check for missing hardware platforms
        missing = [p for p in db.HARDWARE_PLATFORMS if p not in supported]
        
        if not missing:
            results['skipped'] += 1
            continue
        
        # Get the template
        template = db.get_template(model_type, template_type)
        
        if template is None:
            logger.warning(f"Template not found: {model_type}/{template_type}")
            results['skipped'] += 1
            continue
        
        # Validate the template
        valid, error = validate_template(template)
        
        if not valid:
            logger.warning(f"Template not valid: {model_type}/{template_type}")
            results['skipped'] += 1
            continue
        
        # Add hardware-specific variables to the template
        for hardware in missing:
            # Generate hardware-specific template
            hw_template = template
            
            # Update hardware-specific sections
            # Add hardware detection code
            if f"has_{hardware} = " not in hw_template:
                # Add hardware detection to the template
                device_detection = f"""
        # Check for {hardware} support
        has_{hardware} = False
        try:
            # Specific detection for {hardware}
            if {hardware}_detection_code_here:
                has_{hardware} = True
                device = "{hardware}"
        except Exception as e:
            logger.debug(f"No {hardware} support: {{e}}")
"""
                # Insert after other hardware detection
                hw_template = re.sub(
                    r'(# Check for \w+ support.*?logger\.debug.*?\n)',
                    r'\1' + device_detection,
                    hw_template,
                    flags=re.DOTALL
                )
            
            # Add to database
            success = db.add_template(model_type, template_type, hw_template, hardware)
            
            if success:
                logger.info(f"Added {hardware} support to {model_type}/{template_type}")
                results['updated'] += 1
                results['details'].append({
                    'model_type': model_type,
                    'template_type': template_type,
                    'hardware': hardware,
                    'status': 'added'
                })
            else:
                logger.error(f"Failed to add {hardware} support to {model_type}/{template_type}")
                results['failed'] += 1
    
    return results

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Fix template syntax errors and enhance hardware support")
    
    parser.add_argument(
        "--db-path", type=str,
        help="Path to the DuckDB database"
    )
    parser.add_argument(
        "--json-path", type=str,
        help="Path to the JSON template file"
    )
    parser.add_argument(
        "--fix-template", action="store_true",
        help="Fix a specific template"
    )
    parser.add_argument(
        "--model-type", type=str,
        help="Model type for template fix"
    )
    parser.add_argument(
        "--template-type", type=str, default="test",
        help="Template type for template fix"
    )
    parser.add_argument(
        "--hardware", type=str,
        help="Hardware platform for template fix"
    )
    parser.add_argument(
        "--fix-all", action="store_true",
        help="Fix all problematic templates"
    )
    parser.add_argument(
        "--add-hardware-support", action="store_true",
        help="Add missing hardware platform support to templates"
    )
    parser.add_argument(
        "--output", type=str,
        help="Output file for results"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Enable verbose logging"
    )
    
    return parser.parse_args()

def main():
    """Main function."""
    args = parse_args()
    
    # Configure logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")
    
    # Fix a specific template
    if args.fix_template:
        if not args.model_type:
            logger.error("--model-type is required for template fix")
            return 1
        
        # Fix the template
        db = TemplateDatabase(args.db_path, args.json_path)
        template = db.get_template(args.model_type, args.template_type, args.hardware)
        
        if template is None:
            logger.error(f"Template not found: {args.model_type}/{args.template_type}/{args.hardware}")
            return 1
        
        # Validate the template
        valid, error = validate_template(template)
        
        if valid:
            logger.info("Template is already valid")
            return 0
        
        logger.info(f"Template validation error: {error}")
        
        # Fix the template
        fixed_template, fixed_valid, fixed_error = fix_template(template)
        
        if fixed_valid:
            logger.info("Template fixed successfully")
            
            # Update the template in the database
            success = db.add_template(args.model_type, args.template_type, fixed_template, args.hardware)
            
            if success:
                logger.info("Template updated in database")
            else:
                logger.error("Failed to update template in database")
                return 1
        else:
            logger.error(f"Failed to fix template: {fixed_error}")
            return 1
    
    # Fix all problematic templates
    elif args.fix_all:
        logger.info("Fixing all problematic templates")
        
        results = fix_all_templates(args.db_path, args.json_path)
        
        logger.info(f"Total templates: {results['total']}")
        logger.info(f"Fixed templates: {results['fixed']}")
        logger.info(f"Failed templates: {results['failed']}")
        logger.info(f"Skipped templates: {results['skipped']}")
        
        # Output results to file if requested
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
        
        return 0 if results['failed'] == 0 else 1
    
    # Add missing hardware support
    elif args.add_hardware_support:
        logger.info("Adding missing hardware platform support")
        
        results = add_missing_hardware_support(
            args.db_path,
            args.json_path,
            [args.model_type] if args.model_type else None
        )
        
        logger.info(f"Total template groups: {results['total_groups']}")
        logger.info(f"Updated groups: {results['updated']}")
        logger.info(f"Skipped groups: {results['skipped']}")
        logger.info(f"Failed groups: {results['failed']}")
        
        # Output results to file if requested
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
        
        return 0 if results['failed'] == 0 else 1
    
    # No action specified
    else:
        logger.error("No action specified")
        return 1

if __name__ == "__main__":
    sys.exit(main())