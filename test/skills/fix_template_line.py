#!/usr/bin/env python3
"""Fix specific template line issues."""

import os
import re

# Paths
templates_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")

# Templates to fix with specific line numbers and replacements
fixes = [
    {
        "file": "vision_text_template.py",
        "replacements": [
            (776, '        print("\\nVISION TEXT Models Testing Summary:")')
        ]
    },
    {
        "file": "speech_template.py",
        "replacements": [
            (757, '        print("\\nSPEECH Models Testing Summary:")')
        ]
    },
    {
        "file": "multimodal_template.py",
        "replacements": [
            (778, '        print("\\nMULTIMODAL Models Testing Summary:")')
        ]
    }
]

# Fix each template
for fix in fixes:
    template_path = os.path.join(templates_dir, fix["file"])
    
    # Read the template line by line
    with open(template_path, 'r') as f:
        lines = f.readlines()
    
    # Apply replacements
    for line_number, new_line in fix["replacements"]:
        if 0 <= line_number-1 < len(lines):
            lines[line_number-1] = new_line + '\n'
    
    # Write back to a new file
    fixed_path = template_path + ".direct_fix"
    with open(fixed_path, 'w') as f:
        f.write(''.join(lines))
    
    print(f"Fixed {fix['file']} with direct line replacements")
    
    # Verify
    try:
        import py_compile
        py_compile.compile(fixed_path)
        print(f"✅ {fixed_path} compiles successfully!")
        
        # Replace original
        os.replace(fixed_path, template_path)
        print(f"✅ Replaced {template_path} with fixed version")
    except Exception as e:
        print(f"❌ {fixed_path} has syntax errors: {e}")

print("Done!")