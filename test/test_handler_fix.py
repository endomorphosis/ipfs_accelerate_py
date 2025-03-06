#!/usr/bin/env python3
"""
Script to fix MockHandler implementation in generator files.
"""

import re
import os
import shutil
from datetime import datetime

def backup_file(file_path):
    """Create backup of the file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = f"{file_path}.bak_{timestamp}"
    print(f"Creating backup: {backup_path}")
    shutil.copy2(file_path, backup_path)
    return backup_path

def fix_mock_handler(file_path):
    """Fix MockHandler implementation."""
    print(f"Fixing MockHandler in {file_path}")
    
    # Create backup
    backup_file(file_path)
    
    # Read file
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Find all MockHandler classes
    mock_handler_pattern = r'class MockHandler:.*?def __call__\(self, \*args, \*\*kwargs\):.*?return \{.*?\}'
    
    # Replace with corrected implementation
    correct_mock_handler = '''class MockHandler:
    """Mock handler for platforms that don't have real implementations."""
    
    def __init__(self, model_path, platform="cpu"):
        self.model_path = model_path
        self.platform = platform
        print(f"Created mock handler for {platform}")
    
    def __call__(self, *args, **kwargs):
        """Return mock output."""
        print(f"MockHandler for {self.platform} called with {len(args)} args and {len(kwargs)} kwargs")
        return {"output": "MOCK OUTPUT", "implementation_type": f"MOCK_{self.platform.upper()}"}'''
    
    # Use regular expression substitution to fix the mock handler
    fixed_content = re.sub(mock_handler_pattern, correct_mock_handler, content, flags=re.DOTALL)
    
    # Write fixed content
    with open(file_path, 'w') as f:
        f.write(fixed_content)
    
    print(f"Fixed MockHandler in {file_path}")
    return True

def main():
    """Main function."""
    # Fix merged_test_generator.py
    if os.path.exists("merged_test_generator.py"):
        fix_mock_handler("merged_test_generator.py")
    
    # Fix fixed_merged_test_generator.py
    if os.path.exists("fixed_merged_test_generator.py"):
        fix_mock_handler("fixed_merged_test_generator.py")
    
    print("Done fixing MockHandler implementations")

if __name__ == "__main__":
    main()