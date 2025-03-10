#!/usr/bin/env python
"""
Script to fix common errors in template files.

This script analyzes template files for common errors and attempts to fix them.
"""

import os
import sys
import argparse
import re
from pathlib import Path
from typing import Dict, List, Any

# Import simple template validator
sys.path.append(str(Path(__file__).parent))
from simple_template_validator import validate_template_file

def fix_indentation_errors(file_path: str) -> bool:
    """
    Fix indentation errors in a template file.
    
    Args:
        file_path: Path to the template file
        
    Returns:
        True if changes were made, False otherwise
    """
    try:
        with open(file_path, 'r') as f:
            content = f.read()
    except (IOError, UnicodeDecodeError) as e:
        print(f"Failed to read file: {str(e)}")
        return False
    
    # Check for common indentation errors
    lines = content.split('\n')
    fixed_lines = []
    changes_made = False
    
    # Process lines to fix indentation
    in_triple_quote = False
    for i, line in enumerate(lines):
        fixed_line = line
        
        # Track triple quotes (simple approach)
        if '"""' in line:
            in_triple_quote = not in_triple_quote
        
        # Fix common issues: unexpected indent
        if re.match(r'^\s+"""', line) and not in_triple_quote:
            # Remove indentation before triple quotes that should be at column 0
            fixed_line = line.lstrip()
            changes_made = True
        
        # Fix incorrect indentation in method definitions
        if i > 0 and re.match(r'^\s+def\s+\w+\(', lines[i-1]) and re.match(r'^\s{8,}', line):
            # Reduce excessive indentation in method bodies
            spaces = len(line) - len(line.lstrip())
            if spaces > 8:
                fixed_line = ' ' * 8 + line.lstrip()
                changes_made = True
        
        fixed_lines.append(fixed_line)
    
    if changes_made:
        # Write fixed content
        try:
            with open(file_path, 'w') as f:
                f.write('\n'.join(fixed_lines))
            print(f"Fixed indentation errors in {file_path}")
            return True
        except IOError as e:
            print(f"Failed to write file: {str(e)}")
            return False
    
    return False

def fix_syntax_errors(file_path: str) -> bool:
    """
    Fix common syntax errors in a template file.
    
    Args:
        file_path: Path to the template file
        
    Returns:
        True if changes were made, False otherwise
    """
    try:
        with open(file_path, 'r') as f:
            content = f.read()
    except (IOError, UnicodeDecodeError) as e:
        print(f"Failed to read file: {str(e)}")
        return False
    
    # Check for common syntax errors
    changes_made = False
    
    # Fix triple-quoted docstrings inside method definitions
    # Look for methods with docstrings as the first line
    pattern = r'def\s+(\w+)\(([^)]*)\):\s*\n\s+"""([^"]*?)"""'
    
    def replace_docstring(match):
        # Get method name and parameters
        method_name = match.group(1)
        params = match.group(2)
        docstring = match.group(3)
        
        # Create replacement with # comment instead of triple quotes
        return f'def {method_name}({params}):\n    # {docstring}'
    
    # Apply the replacement
    new_content = re.sub(pattern, replace_docstring, content)
    if new_content != content:
        content = new_content
        changes_made = True
    
    # Fix missing required imports
    if 'import torch' not in content and 'from torch import' not in content:
        # Add torch import if missing
        import_section = "import os\nimport sys\nimport torch\nimport logging\n"
        if 'import os' in content and 'import sys' in content:
            # Replace existing imports
            content = re.sub(r'import os\s*\nimport sys', import_section, content)
        else:
            # Add imports at the top after any module docstring
            if re.match(r'^\s*""".*?"""\s*\n', content, re.DOTALL):
                content = re.sub(r'(^\s*""".*?"""\s*\n)', r'\1' + import_section, content, flags=re.DOTALL)
            else:
                content = import_section + content
        changes_made = True
    
    if changes_made:
        # Write fixed content
        try:
            with open(file_path, 'w') as f:
                f.write(content)
            print(f"Fixed syntax errors in {file_path}")
            return True
        except IOError as e:
            print(f"Failed to write file: {str(e)}")
            return False
    
    return False

def fix_hardware_support(file_path: str) -> bool:
    """
    Fix missing hardware support in a template file.
    
    Args:
        file_path: Path to the template file
        
    Returns:
        True if changes were made, False otherwise
    """
    try:
        with open(file_path, 'r') as f:
            content = f.read()
    except (IOError, UnicodeDecodeError) as e:
        print(f"Failed to read file: {str(e)}")
        return False
    
    # Check for missing hardware support
    changes_made = False
    
    # Detect hardware platforms mentioned in the file
    hardware_platforms = ['cpu', 'cuda', 'mps', 'rocm', 'openvino', 'webnn', 'webgpu']
    missing_platforms = []
    
    for platform in hardware_platforms:
        pattern = fr'\b{platform}\b'
        if not re.search(pattern, content, re.IGNORECASE):
            missing_platforms.append(platform)
    
    if missing_platforms:
        # Add missing platform initialization methods
        class_match = re.search(r'class\s+(\w+).*?:', content)
        if class_match:
            class_name = class_match.group(1)
            
            # Generate platform initialization methods
            for platform in missing_platforms:
                platform_lower = platform.lower()
                platform_upper = platform.upper()
                
                init_method = f"""
    def init_{platform_lower}(self):
        \"\"\"Initialize for {platform_upper} platform.\"\"\"
        self.platform = "{platform_upper}"
        self.device = "{platform_lower}"
        return self.load_tokenizer()
                
    def create_{platform_lower}_handler(self):
        \"\"\"Create handler for {platform_upper} platform.\"\"\"
        try:
            # Implementation for {platform_upper}
            return MockHandler(self.model_path, "{platform_lower}")
        except Exception as e:
            print(f"Error creating {platform_upper} handler: {{e}}")
            return MockHandler(self.model_path, "{platform_lower}")
"""
                
                # Find where to insert the new methods
                last_method_match = re.search(r'def\s+create_\w+_handler.*?(?=\n\n|\n\s*$)', content, re.DOTALL)
                if last_method_match:
                    # Insert after the last handler method
                    end_pos = last_method_match.end()
                    content = content[:end_pos] + init_method + content[end_pos:]
                    changes_made = True
                else:
                    # If no handler methods found, try to insert at the end of the class
                    class_match = re.search(r'class\s+' + re.escape(class_name) + r'.*?(?=\n\s*class|\n\s*def|\Z)', content, re.DOTALL)
                    if class_match:
                        end_pos = class_match.end()
                        content = content[:end_pos] + init_method + content[end_pos:]
                        changes_made = True
    
    if changes_made:
        # Write fixed content
        try:
            with open(file_path, 'w') as f:
                f.write(content)
            print(f"Fixed hardware support in {file_path}")
            return True
        except IOError as e:
            print(f"Failed to write file: {str(e)}")
            return False
    
    return False

def fix_template_file(file_path: str) -> Dict[str, Any]:
    """
    Fix common errors in a template file.
    
    Args:
        file_path: Path to the template file
        
    Returns:
        Dictionary with validation results after fixes
    """
    print(f"Checking template: {file_path}")
    
    # Initial validation
    initial_result = validate_template_file(file_path)
    
    if initial_result['valid']:
        print(f"Template is already valid. No fixes needed.")
        return initial_result
    
    print(f"Found {len(initial_result['errors'])} errors. Attempting to fix...")
    
    # Apply fixes
    fixes_applied = []
    
    # Fix indentation errors
    if fix_indentation_errors(file_path):
        fixes_applied.append("Indentation errors fixed")
    
    # Fix syntax errors
    if fix_syntax_errors(file_path):
        fixes_applied.append("Syntax errors fixed")
    
    # Fix hardware support
    if fix_hardware_support(file_path):
        fixes_applied.append("Hardware support fixed")
    
    # Validate again after fixes
    final_result = validate_template_file(file_path)
    
    # Report results
    if not fixes_applied:
        print("No fixes were applied.")
    else:
        print(f"Applied {len(fixes_applied)} fixes:")
        for fix in fixes_applied:
            print(f"  - {fix}")
    
    if final_result['valid']:
        print(f"✅ Template is now valid!")
    else:
        print(f"❌ Template still has {len(final_result['errors'])} errors:")
        for error in final_result['errors'][:5]:
            print(f"  - {error}")
        if len(final_result['errors']) > 5:
            print(f"  - ... and {len(final_result['errors']) - 5} more errors")
    
    final_result['fixes_applied'] = fixes_applied
    return final_result

def fix_templates_in_directory(directory_path: str) -> Dict[str, Dict[str, Any]]:
    """
    Fix all templates in a directory.
    
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
        
        # Fix the template
        result = fix_template_file(str(file_path))
        results[file_path.name] = result
    
    return results

def main():
    """Main function for standalone usage"""
    parser = argparse.ArgumentParser(description="Template Error Fixer")
    parser.add_argument("--file", type=str, help="Fix a single template file")
    parser.add_argument("--dir", type=str, help="Fix all templates in a directory")
    parser.add_argument("--backup", action="store_true", help="Create backups before fixing")
    args = parser.parse_args()
    
    # Create backups if requested
    if args.backup:
        if args.file:
            backup_path = args.file + '.bak'
            try:
                with open(args.file, 'r') as src, open(backup_path, 'w') as dst:
                    dst.write(src.read())
                print(f"Created backup: {backup_path}")
            except Exception as e:
                print(f"Failed to create backup: {e}")
        elif args.dir:
            for file_path in Path(args.dir).glob('*.py'):
                if file_path.name.startswith('_'):
                    continue
                if not (file_path.name.startswith('template_') or file_path.name.endswith('template.py')):
                    continue
                
                backup_path = str(file_path) + '.bak'
                try:
                    with open(file_path, 'r') as src, open(backup_path, 'w') as dst:
                        dst.write(src.read())
                    print(f"Created backup: {backup_path}")
                except Exception as e:
                    print(f"Failed to create backup for {file_path}: {e}")
    
    if args.file:
        # Fix a single file
        fix_template_file(args.file)
    
    elif args.dir:
        # Fix all templates in a directory
        results = fix_templates_in_directory(args.dir)
        
        # Count valid and invalid templates
        valid_count = sum(1 for r in results.values() if r.get('valid', False))
        invalid_count = len(results) - valid_count
        fixed_count = sum(1 for r in results.values() if r.get('fixes_applied', []))
        
        print(f"\nSummary: {fixed_count} templates fixed, {valid_count} valid, {invalid_count} still invalid")
    else:
        parser.print_help()

if __name__ == "__main__":
    main()