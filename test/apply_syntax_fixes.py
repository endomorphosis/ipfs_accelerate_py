#!/usr/bin/env python3
"""
Applies syntax fixes to files with errors.
"""

import os
import shutil
import subprocess
from pathlib import Path

# Map of original file paths to their fixed versions
FIXED_FILES = {
    "../duckdb_api/schema/update_template_database_for_qualcomm.py": "../duckdb_api/schema/fixed/update_template_database_for_qualcomm_fixed.py",
}

def backup_file(file_path):
    """Create a backup of a file."""
    backup_path = f"{file_path}.bak_syntax_fix"
    shutil.copy2(file_path, backup_path)
    print(f"Created backup: {backup_path}")
    return backup_path

def apply_fixes():
    """Apply fixes to files with syntax errors."""
    fixed_count = 0
    
    for original_file, fixed_file in FIXED_FILES.items():
        if os.path.exists(original_file) and os.path.exists(fixed_file):
            try:
                # Create backup of original file
                backup_file(original_file)
                
                # Copy fixed file to original location
                shutil.copy2(fixed_file, original_file)
                print(f"Applied fix: {original_file}")
                fixed_count += 1
            except Exception as e:
                print(f"Error applying fix to {original_file}: {e}")
    
    return fixed_count

def check_syntax(file_path):
    """Check if a file has syntax errors."""
    try:
        result = subprocess.run(
            ["/usr/bin/env", "python", "-m", "py_compile", file_path],
            capture_output=True,
            text=True,
            check=False
        )
        return result.returncode == 0
    except Exception:
        return False

def main():
    """Main function."""
    print("Applying syntax fixes...")
    fixed_count = apply_fixes()
    
    print(f"\nApplied fixes to {fixed_count} files.")
    
    # Check if all fixes worked
    errors = []
    for original_file, _ in FIXED_FILES.items():
        if not check_syntax(original_file):
            errors.append(original_file)
    
    if errors:
        print("\nWarning: Some files still have syntax errors:")
        for file in errors:
            print(f"  - {file}")
    else:
        print("\nAll applied fixes were successful!")
        print("Run fix_syntax_errors.py to check for any remaining syntax errors.")

if __name__ == "__main__":
    main()