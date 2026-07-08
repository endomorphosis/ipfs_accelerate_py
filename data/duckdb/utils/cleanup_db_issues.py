#!/usr/bin/env python3

import os
import sys
import re
import json
import time
import traceback
from datetime import datetime

"""
This script identifies and fixes database integration issues in the test_ipfs_accelerate.py file.
It can detect and clean up lingering SQL fragments and other issues.
"""

def main():
    print("Starting database cleanup...")
    
    # Define file path
    file_path = os.path.join(os.path.dirname(__file__), "test_ipfs_accelerate.py")
    
    if not os.path.exists(file_path):
        print(f"Error: File not found: {file_path}")
        return False
    
    # Read file content
    try:
        with open(file_path, "r") as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading file: {e}")
        return False
    
    # Make a backup
    backup_path = f"{file_path}.bak_{int(time.time())}"
    try:
        with open(backup_path, "w") as f:
            f.write(content)
        print(f"Created backup at {backup_path}")
    except Exception as e:
        print(f"Warning: Failed to create backup: {e}")
    
    # Find and fix JSON/SQL fragment issues
    print("Finding and fixing JSON/SQL fragment issues...")
    
    # Common patterns for SQL fragments that might be left in the file
    patterns = [
        r'return json\.dumps\(.*?\)\s+[a-zA-Z_]+\s+[A-Z]+'  # SQL after JSON dumps
    ]
    
    # Custom fixes for known issues
    known_issues = {
        # Fix for SQL fragments after json.dumps
        r'return json\.dumps\(report_data, indent=2\)([\s\S]+?)(?=def|class|if __name__|$)': 
            'return json.dumps(report_data, indent=2)\n'
    }
    
    # Apply fixes for known issues
    fixed_content = content
    for pattern, replacement in known_issues.items():
        fixed_content = re.sub(pattern, replacement, fixed_content)
    
    # Check if any changes were made
    if fixed_content != content:
        try:
            with open(file_path, "w") as f:
                f.write(fixed_content)
            print("Successfully fixed issues in the file.")
        except Exception as e:
            print(f"Error writing to file: {e}")
            return False
    else:
        print("No issues found in the file.")
    
    # Create a verification file to document the changes
    verification_path = os.path.join(os.path.dirname(__file__), "DB_CLEANUP_VERIFICATION.md")
    try:
        with open(verification_path, "w") as f:
            f.write("# Database Cleanup Verification\n\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("## Actions Performed\n\n")
            f.write("1. Created backup of test_ipfs_accelerate.py\n")
            f.write("2. Scanned for SQL fragments and other database issues\n")
            f.write("3. Applied fixes for known issues\n\n")
            f.write("## Results\n\n")
            if fixed_content != content:
                f.write("Issues were found and fixed successfully.\n")
            else:
                f.write("No issues were found in the file.\n")
        print(f"Created verification file: {verification_path}")
    except Exception as e:
        print(f"Warning: Failed to create verification file: {e}")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)