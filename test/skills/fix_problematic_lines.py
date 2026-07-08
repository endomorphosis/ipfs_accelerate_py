#!/usr/bin/env python3
"""Fix problematic lines in all template files."""

import os
import fileinput
import re

# Path to templates directory
templates_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")
templates = ['vision_text_template.py', 'multimodal_template.py', 'speech_template.py']

def fix_problems_in_file(filepath):
    """Fix problematic print statements in a file."""
    print(f"Processing {filepath}...")
    
    # Create a safe backup
    backup_path = filepath + '.problematic.bak'
    with open(filepath, 'r') as src, open(backup_path, 'w') as dst:
        dst.write(src.read())
    print(f"Created backup at {backup_path}")
    
    # Get lines from the original file
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    # Modified content
    modified_lines = []
    i = 0
    
    # Scan for problematic lines
    while i < len(lines):
        line = lines[i]
        
        # Handle problematic cases
        if 'print("' in line and not line.strip().endswith('"') and not line.strip().endswith('")'):
            # This is likely a problematic line
            if i+1 < len(lines) and lines[i+1].strip().endswith('")'):
                # This is a multiline print that needs fixing
                # Replace with a proper f-string print 
                # For example: print("\nBERT -> print(f"\nBERT Models Testing Summary:")
                
                multiline_text = line.strip()[7:-1] + ' ' + lines[i+1].strip()
                fixed_line = f'        print(f"{multiline_text}'
                modified_lines.append(fixed_line + '\n')
                i += 2  # Skip the next line as we've handled it
                continue
        
        # Keep the line as is
        modified_lines.append(line)
        i += 1
    
    # Write modified content
    with open(filepath, 'w') as f:
        f.writelines(modified_lines)
    
    # Verify the file
    try:
        import py_compile
        py_compile.compile(filepath)
        print(f"✅ Successfully fixed {filepath}")
        return True
    except SyntaxError as e:
        print(f"❌ Syntax error in {filepath}: {e}")
        
        # Restore from backup
        os.replace(backup_path, filepath)
        print(f"Restored from backup {backup_path}")
        return False

def fix_file_manually(filepath, line_num, replacement):
    """Fix a specific line in a file with a direct replacement."""
    print(f"Manual fix for {filepath} at line {line_num}")
    
    # Create a safe backup
    backup_path = filepath + '.manual.bak'
    with open(filepath, 'r') as src, open(backup_path, 'w') as dst:
        dst.write(src.read())
    print(f"Created backup at {backup_path}")
    
    # Read all lines
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    # Replace the specific line
    if 0 <= line_num - 1 < len(lines):
        lines[line_num - 1] = replacement + '\n'
    
    # Write back
    with open(filepath, 'w') as f:
        f.writelines(lines)
    
    # Verify the file
    try:
        import py_compile
        py_compile.compile(filepath)
        print(f"✅ Successfully fixed {filepath}")
        return True
    except SyntaxError as e:
        print(f"❌ Syntax error in {filepath}: {e}")
        
        # Restore from backup
        os.replace(backup_path, filepath)
        print(f"Restored from backup {backup_path}")
        return False

def main():
    """Main entry point."""
    manual_fixes = [
        # vision_text_template.py
        (os.path.join(templates_dir, 'vision_text_template.py'), 776, '        print("\nVISION TEXT Models Testing Summary:")'),
        
        # multimodal_template.py
        (os.path.join(templates_dir, 'multimodal_template.py'), 778, '        print("\nMULTIMODAL Models Testing Summary:")'),
        
        # speech_template.py
        (os.path.join(templates_dir, 'speech_template.py'), 757, '        print("\nSPEECH Models Testing Summary:")'),
    ]
    
    # Try general approach first
    print("Attempting automatic fixes for all template files...")
    for template in templates:
        filepath = os.path.join(templates_dir, template)
        fix_problems_in_file(filepath)
    
    # Apply manual fixes if needed
    print("\nApplying specific manual fixes...")
    for filepath, line_num, replacement in manual_fixes:
        fix_file_manually(filepath, line_num, replacement)
    
    print("\nAll fixes completed!")

if __name__ == "__main__":
    main()