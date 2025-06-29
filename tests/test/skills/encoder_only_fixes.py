#!/usr/bin/env python3
"""Fix encoder_only_template.py file."""

import os
import shutil

# Template to fix
template_path = "templates/encoder_only_template.py"

def fix_encoder_only_template():
    """Fix syntax errors in encoder_only_template.py."""
    # Create backup
    backup_path = template_path + ".syntax.bak"
    shutil.copy2(template_path, backup_path)
    print(f"Created backup: {backup_path}")
    
    # Read the file
    with open(template_path, 'r') as f:
        lines = f.readlines()
    
    # Fix specific problematic lines
    for i, line in enumerate(lines):
        # Fix print with parenthesis issue on line 683
        if i == 682:  # Line 683 (0-indexed)
            if 'print(f"\\n" + "="*50)' in line:
                lines[i] = '    print(f"\\n" + "="*50)\n'
    
    # Write the fixed content
    with open(template_path, 'w') as f:
        f.writelines(lines)
    
    # Verify syntax
    try:
        import py_compile
        py_compile.compile(template_path)
        print(f"✅ Successfully fixed {template_path}")
        return True
    except Exception as e:
        print(f"❌ Syntax error in {template_path}: {e}")
        # Restore from backup
        shutil.copy2(backup_path, template_path)
        print(f"Restored from backup")
        return False

if __name__ == "__main__":
    fix_encoder_only_template()