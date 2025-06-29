#!/usr/bin/env python3
"""Create new fixed versions of problematic template files."""

import os
import sys
import shutil

# Path to templates directory
templates_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")

def create_fixed_vision_text_template():
    """Create a fixed version of vision_text_template.py."""
    source_path = os.path.join(templates_dir, "vision_text_template.py")
    temp_path = os.path.join(templates_dir, "vision_text_template.py.new")
    
    # Create backup
    backup_path = source_path + ".final.bak"
    shutil.copy2(source_path, backup_path)
    print(f"Created backup: {backup_path}")
    
    # Fix specific lines
    with open(source_path, 'r') as src:
        lines = src.readlines()
    
    # Fix problematic lines at line 761
    if len(lines) >= 762:
        lines[760] = '        print("\\nVISION TEXT Models Testing Summary:")\n'
        lines[761] = '        total = len(results)\n'
    
    # Write fixed file
    with open(temp_path, 'w') as dst:
        dst.writelines(lines)
    
    # Verify syntax
    try:
        import py_compile
        py_compile.compile(temp_path)
        print(f"✅ {temp_path} successfully created with valid syntax")
        # Replace original
        shutil.move(temp_path, source_path)
        print(f"✅ Replaced {source_path} with fixed version")
        return True
    except Exception as e:
        print(f"❌ Syntax error in {temp_path}: {e}")
        # Remove temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return False

def create_fixed_speech_template():
    """Create a fixed version of speech_template.py."""
    source_path = os.path.join(templates_dir, "speech_template.py")
    temp_path = os.path.join(templates_dir, "speech_template.py.new")
    
    # Create backup
    backup_path = source_path + ".final.bak"
    shutil.copy2(source_path, backup_path)
    print(f"Created backup: {backup_path}")
    
    # Fix specific lines
    with open(source_path, 'r') as src:
        lines = src.readlines()
    
    # Fix problematic lines at line 742
    if len(lines) >= 743:
        lines[741] = '        print("\\nSPEECH Models Testing Summary:")\n'
        lines[742] = '        for model in models:\n'
    
    # Write fixed file
    with open(temp_path, 'w') as dst:
        dst.writelines(lines)
    
    # Verify syntax
    try:
        import py_compile
        py_compile.compile(temp_path)
        print(f"✅ {temp_path} successfully created with valid syntax")
        # Replace original
        shutil.move(temp_path, source_path)
        print(f"✅ Replaced {source_path} with fixed version")
        return True
    except Exception as e:
        print(f"❌ Syntax error in {temp_path}: {e}")
        # Remove temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return False

def create_fixed_multimodal_template():
    """Create a fixed version of multimodal_template.py."""
    source_path = os.path.join(templates_dir, "multimodal_template.py")
    temp_path = os.path.join(templates_dir, "multimodal_template.py.new")
    
    # Create backup
    backup_path = source_path + ".final.bak"
    shutil.copy2(source_path, backup_path)
    print(f"Created backup: {backup_path}")
    
    # Fix specific lines
    with open(source_path, 'r') as src:
        lines = src.readlines()
    
    # Fix problematic lines at line 763
    if len(lines) >= 764:
        lines[762] = '        print("\\nMULTIMODAL Models Testing Summary:")\n'
        lines[763] = '        for model in models:\n'
    
    # Write fixed file
    with open(temp_path, 'w') as dst:
        dst.writelines(lines)
    
    # Verify syntax
    try:
        import py_compile
        py_compile.compile(temp_path)
        print(f"✅ {temp_path} successfully created with valid syntax")
        # Replace original
        shutil.move(temp_path, source_path)
        print(f"✅ Replaced {source_path} with fixed version")
        return True
    except Exception as e:
        print(f"❌ Syntax error in {temp_path}: {e}")
        # Remove temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return False

def main():
    """Create fixed versions of all problematic templates."""
    results = []
    
    print("=== Creating fixed version of vision_text_template.py ===")
    results.append(create_fixed_vision_text_template())
    
    print("\n=== Creating fixed version of speech_template.py ===")
    results.append(create_fixed_speech_template())
    
    print("\n=== Creating fixed version of multimodal_template.py ===")
    results.append(create_fixed_multimodal_template())
    
    success = all(results)
    print(f"\nFixed templates: {sum(results)} of {len(results)}")
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())