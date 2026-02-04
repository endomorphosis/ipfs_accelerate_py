#!/usr/bin/env python3
"""Direct line fixes for template files."""

import os
import re
import shutil

# Templates to fix
fixes = [
    # vision_text_template.py
    {"file": "templates/vision_text_template.py", "lines": {
        775: '        # Print summary',
        776: '        print(f"\\nVISION TEXT Models Testing Summary:")',
        777: '        total = len(results)'
    }},
    # speech_template.py
    {"file": "templates/speech_template.py", "lines": {
        756: '        # Print summary',
        757: '        print(f"\\nSPEECH Models Testing Summary:")',
        758: '        total = len(results)'
    }},
    # multimodal_template.py
    {"file": "templates/multimodal_template.py", "lines": {
        777: '        # Print summary',
        778: '        print(f"\\nMULTIMODAL Models Testing Summary:")',
        779: '        total = len(results)'
    }}
]

def fix_template(fix_info):
    """Fix template by replacing specific lines."""
    file_path = fix_info["file"]
    
    # Create backup
    backup_path = file_path + ".direct_bak"
    if os.path.exists(file_path):
        shutil.copy2(file_path, backup_path)
        print(f"Created backup: {backup_path}")
    else:
        print(f"File not found: {file_path}")
        return False
    
    try:
        # Read all lines
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        # Apply line replacements
        for line_num, new_line in fix_info["lines"].items():
            if 0 <= line_num-1 < len(lines):
                lines[line_num-1] = new_line + '\n'
        
        # Write fixed content
        with open(file_path, 'w') as f:
            f.writelines(lines)
        
        # Verify fix
        try:
            import py_compile
            py_compile.compile(file_path)
            print(f"✅ Fixed {file_path}")
            return True
        except Exception as e:
            print(f"❌ Syntax error in {file_path}: {e}")
            # Restore from backup
            shutil.copy2(backup_path, file_path)
            print(f"Restored from backup")
            return False
    except Exception as e:
        print(f"Error fixing {file_path}: {e}")
        # Restore from backup if exists
        if os.path.exists(backup_path):
            shutil.copy2(backup_path, file_path)
            print(f"Restored from backup")
        return False

def main():
    """Apply all direct line fixes."""
    success_count = 0
    for fix_info in fixes:
        if fix_template(fix_info):
            success_count += 1
    
    print(f"Fixed {success_count} of {len(fixes)} templates")

if __name__ == "__main__":
    main()