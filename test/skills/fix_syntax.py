#!/usr/bin/env python3
"""
Fix basic syntax issues in Python test files.

This script applies very basic indentation rules to make the file at least syntactically valid.
It doesn't aim for perfect PEP 8 compliance, just enough to compile.

Usage:
    python fix_syntax.py <file_path>
"""

import sys
import os
import re
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def fix_syntax(file_path, backup=True):
    """Fix basic syntax issues in a Python file."""
    try:
        # Read the file content
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Create backup if requested
        if backup:
            backup_path = f"{file_path}.bak"
            with open(backup_path, 'w') as f:
                f.write(content)
            logger.info(f"Created backup at {backup_path}")
        
        # Simple replacements to fix obvious syntax errors
        
        # 1. Fix class definition indent - must be at column 0
        content = re.sub(r'^\s+class\s+(\w+):', r'class \1:', content, flags=re.MULTILINE)
        
        # 2. Fix method definition indent - must be 4 spaces
        content = re.sub(r'^\s*def\s+(\w+)\(self', r'    def \1(self', content, flags=re.MULTILINE)
        
        # 3. Fix function definition indent - must be at column 0
        content = re.sub(r'^\s+def\s+(\w+)\((?!self)', r'def \1(', content, flags=re.MULTILINE)
        
        # 4. Fix docstring indentation - must be 4 spaces more than containing block
        content = re.sub(r'^\s{1,3}(""".*?""")', r'    \1', content, flags=re.MULTILINE)
        
        # 5. Fix method body indentation - minimum 8 spaces
        lines = content.split('\n')
        in_method = False
        fixed_lines = []
        
        for line in lines:
            stripped = line.strip()
            
            # Skip empty lines
            if not stripped:
                fixed_lines.append(line)
                continue
            
            # Check for method definition
            if stripped.startswith('def ') and 'self' in stripped:
                in_method = True
                fixed_lines.append(line)
            # Check for class definition or function definition (exit method context)
            elif stripped.startswith(('class ', 'def ')) and 'self' not in stripped:
                in_method = False
                fixed_lines.append(line)
            # Fix method body indentation
            elif in_method and not line.startswith('    def '):
                # Already has 8+ spaces, keep as is
                if line.startswith('        '):
                    fixed_lines.append(line)
                # Has 4-7 spaces, increase to 8
                elif line.startswith('    '):
                    fixed_lines.append('        ' + line.lstrip())
                # Less than 4 spaces, set to 8
                else:
                    fixed_lines.append('        ' + stripped)
            # Keep other lines as is
            else:
                fixed_lines.append(line)
        
        # Join lines back together
        content = '\n'.join(fixed_lines)
        
        # Write fixed content
        with open(file_path, 'w') as f:
            f.write(content)
        
        # Verify syntax
        try:
            compile(content, file_path, 'exec')
            logger.info(f"✅ Fixed syntax in {file_path}")
            return True
        except SyntaxError as e:
            logger.error(f"❌ Syntax errors remain in {file_path}: {e}")
            logger.error(f"  Line {e.lineno}, column {e.offset}: {e.text.strip() if e.text else ''}")
            return False
        
    except Exception as e:
        logger.error(f"❌ Error fixing syntax in {file_path}: {e}")
        return False

def main():
    if len(sys.argv) < 2:
        logger.error("Usage: python fix_syntax.py <file_path>")
        return 1
    
    file_path = sys.argv[1]
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return 1
    
    # Fix syntax
    if fix_syntax(file_path):
        return 0
    else:
        return 1

if __name__ == "__main__":
    sys.exit(main())