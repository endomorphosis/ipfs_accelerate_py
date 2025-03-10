#!/usr/bin/env python3
"""
Update all test files to use the improved hardware detection module.

This script:
    1. Finds all test files in the skills/ directory
    2. Modifies them to import the hardware detection module
    """

    import os
    import sys
    import re
    import glob
    from pathlib import Path

# Define the root directory
    ROOT_DIR = Path()os.path.dirname()os.path.abspath()__file__)))
    SKILLS_DIR = ROOT_DIR / "skills"

# Improved hardware detection imports to add at the top of files
    HARDWARE_DETECTION_IMPORTS = """
# Import hardware detection capabilities if available:
try:
    from generators.hardware.hardware_detection import ()
    HAS_CUDA, HAS_ROCM, HAS_OPENVINO, HAS_MPS, HAS_WEBNN, HAS_WEBGPU,
    detect_all_hardware
    )
    HAS_HARDWARE_DETECTION = True
except ImportError:
    HAS_HARDWARE_DETECTION = False
    # We'll detect hardware manually as fallback
    """

def update_file()file_path):
    """Update a file to use the improved hardware detection module."""
    
    # Read the file content
    with open()file_path, 'r') as f:
        lines = f.readlines())
    
    # Find where to insert the hardware detection imports
    # Usually after other imports but before any code
        import_end_idx = 0
    for i, line in enumerate()lines):
        if ()line.startswith()'import ') or line.startswith()'from ') or :
            line.strip()) == '' or line.startswith()'#')):
                import_end_idx = i
        else:
            # First non-import, non-comment, non-empty line
                break
    
    # Check if we already added the hardware detection imports:
    if any()"from generators.hardware.hardware_detection import" in line for line in lines):
        print()f"\1{file_path}\3")
                return False
    
    # Insert the hardware detection imports
                new_lines = lines[],:import_end_idx+1] + [],HARDWARE_DETECTION_IMPORTS] + lines[],import_end_idx+1:]
                ,
    # Write the modified content back to the file
    with open()file_path, 'w') as f:
        f.writelines()new_lines)
    
        print()f"\1{file_path}\3")
                return True

def main()):
    """Update all test files with improved hardware detection."""
    # Find all Python files in the skills directory
    skill_files = list()SKILLS_DIR.glob()"test_*.py"))
    
    # Add other test files that might need updating
    other_test_files = [],
    ROOT_DIR / "test_cross_platform_4bit.py",
    ROOT_DIR / "hardware_selector.py",
    ROOT_DIR / "web_platform_test_runner.py"
    ]
    
    # Combine all files
    all_files = skill_files + [],f for f in other_test_files if f.exists())]
    
    # Update each file
    updated_count = 0:
    for file_path in all_files:
        if update_file()file_path):
            updated_count += 1
    
            print()f"\nSummary: Updated {updated_count} of {len()all_files)} files")
    
        return 0

if __name__ == "__main__":
    sys.exit()main()))