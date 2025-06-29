#!/usr/bin/env python3
"""
Fix syntax errors in a test file.

This script focuses specifically on fixing syntax errors like unterminated strings.
"""

import os
import sys
import re
import logging
import argparse

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

def fix_unterminated_strings(file_path):
    """Fix syntax errors in the file."""
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        # Create backup
        backup_path = f"{file_path}.syntax.bak"
        with open(backup_path, 'w') as f:
            f.writelines(lines)
        logger.info(f"Created backup at: {backup_path}")
        
        # Look for problematic lines
        fixed_lines = []
        in_triple_quote = False
        
        for i, line in enumerate(lines):
            # Check for problematic triple quotes (""""" or '''')
            if '""""' in line:
                logger.info(f"Fixing extra quotes on line {i+1}")
                line = line.replace('""""', '"""')
            if "''''" in line:
                logger.info(f"Fixing extra quotes on line {i+1}")
                line = line.replace("''''", "'''")
            
            # Fix extra parentheses, including in import statements
            if ')))' in line:
                logger.info(f"Fixing extra parentheses on line {i+1}")
                # Fix import statements separately to handle the special case
                if 'import' in line and '(' in line:
                    line = re.sub(r'import\s*\([)]+', 'import (', line)
                else:
                    line = re.sub(r'\){3,}', ')', line)
            
            # Fix extra braces
            if '}}}}}}' in line:
                logger.info(f"Fixing extra braces on line {i+1}")
                line = re.sub(r'\}{3,}', '}', line)
            
            # Fix brackets + commas
            if '[],(' in line:
                logger.info(f"Fixing bracket+comma syntax on line {i+1}")
                line = re.sub(r'\[\],\(', '[', line)
                line = re.sub(r'\[\],([^,\]]+)\]', r'[\1]', line)
            
            # Fix commas in dict literals
            if '{},(' in line:
                logger.info(f"Fixing curly brace+comma syntax on line {i+1}")
                line = re.sub(r'\{\},\(', '{', line)
                line = re.sub(r'\{\},([^,\}]+)\}', r'{\1}', line)
            
            # Fix invalid Python identifiers with hyphens
            if 'KOSMOS-2' in line:
                logger.info(f"Fixing invalid Python identifier on line {i+1}")
                line = line.replace('KOSMOS-2', 'KOSMOS_2')
            
            # Fix improper try statements
            if 'try::' in line:
                logger.info(f"Fixing try statement on line {i+1}")
                line = line.replace('try::', 'try:')
            
            # Check for odd number of quotes
            if line.count('"""') % 2 == 1:
                if not in_triple_quote:
                    in_triple_quote = True
                else:
                    in_triple_quote = False
            
            # Fix unterminated strings (not inside triple quotes)
            if not in_triple_quote:
                if line.count('"') % 2 == 1 and not line.strip().startswith('#'):
                    logger.info(f"Fixing unterminated double quote on line {i+1}")
                    line = line.rstrip() + '"\n'
                
                if line.count("'") % 2 == 1 and not line.strip().startswith('#'):
                    logger.info(f"Fixing unterminated single quote on line {i+1}")
                    line = line.rstrip() + "'\n"
            
            fixed_lines.append(line)
        
        # Add closing triple quote if needed
        if in_triple_quote:
            logger.info("Adding missing triple quote at end of file")
            fixed_lines.append('"""\n')
        
        # Write fixed content
        with open(file_path, 'w') as f:
            f.writelines(fixed_lines)
        
        # Second pass - fix content as a whole
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Fix import indentation
        content = re.sub(r'^\s+import', 'import', content, flags=re.MULTILINE)
        content = re.sub(r'^\s+from', 'from', content, flags=re.MULTILINE)
        
        # Fix hardware detection import
        content = re.sub(r'from generators\.hardware\.hardware_detection import \(+\)+', 'from generators.hardware.hardware_detection import (', content)
        content = re.sub(r'HAS_CUDA, HAS_ROCM, HAS_OPENVINO, HAS_MPS, HAS_WEBNN, HAS_WEBGPU,', 'HAS_CUDA, HAS_ROCM, HAS_OPENVINO, HAS_MPS, HAS_WEBNN, HAS_WEBGPU,\n    detect_all_hardware', content)
        
        # Fix class declarations with invalid names
        content = re.sub(r'class TestKOSMOS-2Models', 'class TestKosmos2Models', content)
        
        # Fix registry variable names with hyphens
        content = re.sub(r'KOSMOS-2_MODELS_REGISTRY', 'KOSMOS_2_MODELS_REGISTRY', content)
        
        # Write fixed content
        with open(file_path, 'w') as f:
            f.write(content)
        
        # Verify syntax
        with open(file_path, 'r') as f:
            content = f.read()
        compile(content, file_path, 'exec')
        logger.info(f"✅ Syntax is valid after fixes")
        return True
    
    except SyntaxError as e:
        logger.error(f"❌ Syntax error after fixes: {e}")
        # Show the problematic line for debugging
        if hasattr(e, 'lineno') and e.lineno is not None:
            line_no = e.lineno - 1  # 0-based index
            if 0 <= line_no < len(fixed_lines):
                logger.error(f"Problematic line {e.lineno}: {fixed_lines[line_no].rstrip()}")
        
        # Restore from backup
        logger.info("Restoring from backup...")
        with open(backup_path, 'r') as f:
            content = f.read()
        with open(file_path, 'w') as f:
            f.write(content)
        return False
    
    except Exception as e:
        logger.error(f"Error fixing file: {e}")
        return False

def fix_specific_line(file_path, line_number):
    """Fix a specific line in the file."""
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        # Create backup
        backup_path = f"{file_path}.line.bak"
        with open(backup_path, 'w') as f:
            f.writelines(lines)
        logger.info(f"Created backup at: {backup_path}")
        
        # Fix the specific line
        if 1 <= line_number <= len(lines):
            line = lines[line_number-1]
            
            # Check for specific issues
            if '""""' in line:
                logger.info(f"Fixing extra quotes on line {line_number}")
                lines[line_number-1] = line.replace('""""', '"""')
            elif "''''" in line:
                logger.info(f"Fixing extra quotes on line {line_number}")
                lines[line_number-1] = line.replace("''''", "'''")
            else:
                # Remove all quotes and add the correct ones
                stripped = line.strip().rstrip(',')
                if stripped.startswith(('"""', "'''")):
                    logger.info(f"Fixing triple quotes on line {line_number}")
                    lines[line_number-1] = '        """\n'
                elif stripped.startswith(('"', "'")):
                    logger.info(f"Fixing string quotes on line {line_number}")
                    lines[line_number-1] = f'        "{stripped.strip("\'\"")}"'
                else:
                    logger.info(f"Line doesn't seem to have quote issues, replacing with empty string")
                    lines[line_number-1] = '        ""\n'
        else:
            logger.error(f"Line number {line_number} out of range (file has {len(lines)} lines)")
            return False
        
        # Write fixed content
        with open(file_path, 'w') as f:
            f.writelines(lines)
        
        # Verify syntax
        with open(file_path, 'r') as f:
            content = f.read()
        compile(content, file_path, 'exec')
        logger.info(f"✅ Syntax is valid after fixes")
        return True
    
    except SyntaxError as e:
        logger.error(f"❌ Syntax error after fixes: {e}")
        # Restore from backup
        logger.info("Restoring from backup...")
        with open(backup_path, 'r') as f:
            content = f.read()
        with open(file_path, 'w') as f:
            f.write(content)
        return False
    
    except Exception as e:
        logger.error(f"Error fixing file: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Fix syntax errors in test files")
    parser.add_argument("--file", type=str, required=True, help="Path to the file to fix")
    parser.add_argument("--line", type=int, help="Specific line number to fix")
    
    args = parser.parse_args()
    
    if args.line:
        success = fix_specific_line(args.file, args.line)
    else:
        success = fix_unterminated_strings(args.file)
    
    if success:
        print(f"Successfully fixed syntax in {args.file}")
    else:
        print(f"Failed to fix syntax in {args.file}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())