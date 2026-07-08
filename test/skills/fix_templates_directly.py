#!/usr/bin/env python3
"""Fix syntax errors in templates directly by creating new files."""

import os

# Paths
templates_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")

# Templates to fix
templates = [
    "vision_text_template.py",
    "speech_template.py",
    "multimodal_template.py"
]

# Fix each template
for template in templates:
    template_path = os.path.join(templates_dir, template)
    template_fixed_path = os.path.join(templates_dir, f"{template}.fixed")
    
    # Read the template
    with open(template_path, 'r') as f:
        content = f.read()
    
    # Replace problematic line with a fixed version
    if "vision_text" in template:
        content = content.replace('print("\nBERT', 'print("\\nBERT')
    elif "speech" in template:
        content = content.replace('print("\nWAV', 'print("\\nWAV')
    elif "multimodal" in template:
        content = content.replace('print("\nCLIP', 'print("\\nCLIP')
    
    # Save the fixed template
    with open(template_fixed_path, 'w') as f:
        f.write(content)
    
    print(f"Fixed template saved to {template_fixed_path}")
    
    # Try to compile
    import py_compile
    try:
        py_compile.compile(template_fixed_path)
        print(f"✅ {template_fixed_path} compiles successfully!")
        
        # Replace the original with the fixed version
        os.replace(template_fixed_path, template_path)
        print(f"✅ Replaced {template_path} with fixed version")
    except Exception as e:
        print(f"❌ {template_fixed_path} has syntax errors: {e}")

print("Done!")