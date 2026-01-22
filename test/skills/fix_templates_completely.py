#!/usr/bin/env python3
"""Create fixed versions of template files by copying from decoder_only_template.py."""

import os
import re
import shutil

# Path to templates directory
templates_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")

# Templates that have correct syntax
correct_template = os.path.join(templates_dir, "decoder_only_template.py")

# Templates to fix
problematic_templates = [
    {
        "file": "vision_text_template.py",
        "model_display": "VISION TEXT"
    },
    {
        "file": "speech_template.py",
        "model_display": "SPEECH"
    },
    {
        "file": "multimodal_template.py",
        "model_display": "MULTIMODAL"
    }
]

def extract_correct_pattern(template_path):
    """Extract the correct print pattern from the template."""
    with open(template_path, 'r') as f:
        content = f.read()
    
    # Find the correct pattern for printing family names
    match = re.search(r'print\((f?)"\\nAvailable.*?models:"', content)
    if match:
        return match.group(0)
    return None

def fix_template(template_info, correct_pattern):
    """Create a fixed version of the template."""
    source_path = os.path.join(templates_dir, template_info["file"])
    target_path = source_path + ".fixed"
    
    # Create a backup
    backup_path = source_path + ".problematic_bak"
    shutil.copy2(source_path, backup_path)
    print(f"Created backup: {backup_path}")
    
    with open(source_path, 'r') as src:
        content = src.read()
    
    # Replace problematic pattern with correct one
    if correct_pattern:
        # Find the problematic pattern - a print statement followed by a model name and "Models Testing Summary"
        problematic_pattern = re.compile(r'print\(".*?\n.*?Models Testing Summary:"\)', re.DOTALL)
        
        # Create the replacement - using the correct pattern but with this template's model display name
        replacement = correct_pattern.replace("GPT-2", template_info["model_display"])
        
        # Apply the replacement
        fixed_content = problematic_pattern.sub(replacement, content)
        
        # Write the fixed content
        with open(target_path, 'w') as dst:
            dst.write(fixed_content)
        
        # Verify the fixed file
        try:
            import py_compile
            py_compile.compile(target_path)
            print(f"✅ {target_path} compiles correctly")
            
            # Replace the original
            shutil.copy2(target_path, source_path)
            print(f"✅ Replaced {source_path} with fixed version")
            return True
        except Exception as e:
            print(f"❌ {target_path} has syntax errors: {e}")
            return False
    else:
        print(f"❌ Could not find correct pattern in the reference template")
        return False

def fix_line_directly(template_info):
    """Fix the problematic line directly with a string replacement."""
    source_path = os.path.join(templates_dir, template_info["file"])
    
    # Create a backup
    backup_path = source_path + ".direct_bak"
    shutil.copy2(source_path, backup_path)
    print(f"Created backup for direct fix: {backup_path}")
    
    with open(source_path, 'r') as src:
        lines = src.readlines()
    
    # Scan for problematic print pattern
    for i, line in enumerate(lines):
        if 'print("' in line and "\\n" not in line and "\n" in line:
            # Found a likely problematic line
            model_name = template_info["model_display"]
            lines[i] = f'        print(f"\\n{model_name} Models Testing Summary:")\n'
            lines[i+1] = "" # Remove the next line which was part of the unterminated string
            break
    
    # Write the fixed content
    with open(source_path, 'w') as dst:
        dst.writelines(lines)
    
    # Verify the fixed file
    try:
        import py_compile
        py_compile.compile(source_path)
        print(f"✅ Direct fix successful for {source_path}")
        return True
    except Exception as e:
        print(f"❌ Direct fix failed for {source_path}: {e}")
        
        # Restore from backup
        shutil.copy2(backup_path, source_path)
        print(f"Restored from backup {backup_path}")
        return False

def main():
    """Fix all problematic template files."""
    # First try the pattern approach
    correct_pattern = extract_correct_pattern(correct_template)
    if correct_pattern:
        print(f"Found correct pattern: {correct_pattern}")
        
        for template_info in problematic_templates:
            if not fix_template(template_info, correct_pattern):
                # If pattern replacement fails, try direct line fixing
                fix_line_directly(template_info)
    else:
        print("Could not find the correct pattern, using direct line fixes")
        for template_info in problematic_templates:
            fix_line_directly(template_info)
    
    print("All templates processed")

if __name__ == "__main__":
    main()