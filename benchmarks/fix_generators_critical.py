#!/usr/bin/env python3
"""
Critical Fix for Test Generators

This script adds the missing create_hardware_compatibility_matrix function and 
KEY_MODEL_HARDWARE_MAP definition to generator files.
"""

import os
import sys
import json
from pathlib import Path

# Configure paths
SCRIPT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = SCRIPT_DIR.parent
TEST_DIR = PROJECT_ROOT / "test"

# Target files to fix
TARGET_FILES = [
    TEST_DIR / "fixed_merged_test_generator.py",
    TEST_DIR / "merged_test_generator.py"
]

# Hardware map to add
KEY_MODEL_HARDWARE_MAP = {
    "bert": {
        "cpu": "REAL",
        "cuda": "REAL",
        "openvino": "REAL",
        "mps": "REAL",
        "rocm": "REAL",
        "webnn": "REAL",
        "webgpu": "REAL"
    },
    "t5": {
        "cpu": "REAL",
        "cuda": "REAL",
        "openvino": "REAL",
        "mps": "REAL",
        "rocm": "REAL",
        "webnn": "REAL",
        "webgpu": "REAL"
    }
}

# The create_hardware_compatibility_matrix function
COMPATIBILITY_FUNCTION = """
def create_hardware_compatibility_matrix():
    # Create matrix with default values
    compatibility = {
        "hardware": {
            "cpu": {"available": True},
            "cuda": {"available": True},
            "rocm": {"available": False},
            "mps": {"available": False},
            "openvino": {"available": False},
            "webnn": {"available": False},
            "webgpu": {"available": False}
        },
        "categories": {
            "text": {
                "cpu": True,
                "cuda": True,
                "rocm": True,
                "mps": True,
                "openvino": True,
                "webnn": True,
                "webgpu": True
            },
            "vision": {
                "cpu": True,
                "cuda": True,
                "rocm": True,
                "mps": True,
                "openvino": True,
                "webnn": True,
                "webgpu": True
            }
        },
        "models": {}
    }
    
    # Add specific model compatibility
    for model_name, hw_support in KEY_MODEL_HARDWARE_MAP.items():
        compatibility["models"][model_name] = {}
        for hw_type, support_level in hw_support.items():
            compatibility["models"][model_name][hw_type] = support_level != "NONE"
    
    return compatibility
"""

# Hardware map definition to add
HARDWARE_MAP_DEFINITION = f"# Hardware support matrix for key models\nKEY_MODEL_HARDWARE_MAP = {json.dumps(KEY_MODEL_HARDWARE_MAP, indent=4)}\n"

def fix_file(file_path):
    """Add missing functions to a file."""
    if not file_path.exists():
        print(f"File {file_path} not found.")
        return False
    
    try:
        # Read the content
        with open(file_path, 'r') as f:
            content = f.read()
        
        changes_made = False
        
        # Check if KEY_MODEL_HARDWARE_MAP is missing
        if "KEY_MODEL_HARDWARE_MAP" in content and "KEY_MODEL_HARDWARE_MAP =" not in content:
            # Find a suitable insertion point - after imports
            import_section_end = content.find("\n\n", content.find("import "))
            if import_section_end == -1:
                import_section_end = content.find("import ") + 100
            
            # Add hardware map definition
            content = content[:import_section_end] + "\n" + HARDWARE_MAP_DEFINITION + content[import_section_end:]
            changes_made = True
            print(f"Added KEY_MODEL_HARDWARE_MAP to {file_path}")
        
        # Check if create_hardware_compatibility_matrix is missing
        if "create_hardware_compatibility_matrix()" in content and "def create_hardware_compatibility_matrix" not in content:
            # Find a suitable insertion point - before the first function call
            call_pos = content.find("create_hardware_compatibility_matrix()")
            if call_pos != -1:
                func_boundary = content.rfind("\ndef ", 0, call_pos)
                if func_boundary != -1:
                    # Insert before the function that calls it
                    content = content[:func_boundary] + COMPATIBILITY_FUNCTION + content[func_boundary:]
                    changes_made = True
                    print(f"Added create_hardware_compatibility_matrix to {file_path}")
        
        # Write changes if needed
        if changes_made:
            with open(file_path, 'w') as f:
                f.write(content)
            print(f"Successfully updated {file_path}")
            return True
        else:
            print(f"No changes needed for {file_path}")
            return True
    
    except Exception as e:
        print(f"Error fixing {file_path}: {e}")
        return False

def main():
    """Main function."""
    print("Applying critical fixes to test generators...")
    
    success_count = 0
    for file_path in TARGET_FILES:
        if fix_file(file_path):
            success_count += 1
    
    print(f"Successfully fixed {success_count} of {len(TARGET_FILES)} files.")
    
    print("\nTo generate a test for BERT, run:")
    print("python ./fixed_merged_test_generator.py --generate bert")
    
    return 0 if success_count == len(TARGET_FILES) else 1

if __name__ == "__main__":
    sys.exit(main())