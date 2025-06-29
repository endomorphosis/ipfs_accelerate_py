#!/usr/bin/env python3
"""
Fix String Issues in Test Generator Files

This script specifically fixes unterminated strings and other syntax issues
in the test generator files.
"""

import os
import sys
import datetime
from pathlib import Path

# Files to fix
FILES = [
    "fixed_merged_test_generator.py",
    "merged_test_generator.py",
    "integrated_skillset_generator.py"
]

def fix_files():
    """Fix all target files."""
    for filename in FILES:
        file_path = Path(os.path.dirname(os.path.abspath(__file__))) / filename
        
        if not file_path.exists():
            print(f"File not found: {file_path}")
            continue
        
        # Create backup
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = file_path.with_suffix(f".py.bak_{timestamp}")
        
        with open(file_path, 'r', encoding='utf-8') as source_file:
            content = source_file.read()
            with open(backup_path, 'w', encoding='utf-8') as backup_file:
                backup_file.write(content)
        
        print(f"Created backup at {backup_path}")
        
        # Fix template definitions
        lines = content.split('\n')
        fixed_lines = []
        skip_until_new_template = False
        
        for i, line in enumerate(lines):
            if skip_until_new_template:
                if line.strip().startswith('template_database['):
                    skip_until_new_template = False
                    fixed_lines.append(line)
                continue
            
            # Fix templates
            if line.strip().startswith('template_database[') and '"""' in line and not line.strip().endswith('"""'):
                # Make sure the template starts with triple quotes and ends with triple quotes
                fixed_lines.append(line)
                in_triple_quotes = True
                for j in range(i+1, len(lines)):
                    if '"""' in lines[j]:
                        in_triple_quotes = False
                        fixed_lines.append(lines[j])
                        break
                    fixed_lines.append(lines[j])
                
                if in_triple_quotes:
                    # If we didn't find a closing triple quote, add one
                    fixed_lines.append('"""')
                    print(f"Added missing closing triple quotes in {file_path} at line {i}")
                
                skip_until_new_template = True
            else:
                fixed_lines.append(line)
        
        # Fix the MockHandler class definition
        in_mock_handler = False
        in_call_method = False
        call_method_lines = []
        
        fixed_content = []
        i = 0
        while i < len(fixed_lines):
            line = fixed_lines[i]
            
            if line.strip() == 'class MockHandler:':
                in_mock_handler = True
                fixed_content.append(line)
            elif in_mock_handler and line.strip().startswith('def __call__(self'):
                in_call_method = True
                call_method_lines = [line]
                j = i + 1
                while j < len(fixed_lines) and (fixed_lines[j].startswith(' ') or not fixed_lines[j].strip()):
                    call_method_lines.append(fixed_lines[j])
                    j += 1
                
                # Check if the last line is a proper end of method
                if j < len(fixed_lines) and fixed_lines[j].strip().startswith('"""') and not fixed_lines[j].strip().endswith('"""'):
                    # This is a string continuation which is causing the issue
                    # Skip until the next proper line
                    while j < len(fixed_lines) and not (fixed_lines[j].strip().startswith('template_database[') or fixed_lines[j].strip().startswith('def ')):
                        j += 1
                
                # Add a proper method implementation
                fixed_content.append('    def __call__(self, *args, **kwargs):')
                fixed_content.append('        """Return mock output."""')
                fixed_content.append('        print(f"MockHandler for {self.platform} called with {len(args)} args and {len(kwargs)} kwargs")')
                fixed_content.append('        return {"output": "MOCK OUTPUT", "implementation_type": f"MOCK_{self.platform.upper()}"}')
                
                i = j - 1 if j < len(fixed_lines) else len(fixed_lines) - 1
            else:
                fixed_content.append(line)
            
            i += 1
        
        # Write fixed content
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(fixed_content))
        
        print(f"Fixed string issues in {file_path}")

if __name__ == "__main__":
    fix_files()
    print("All files fixed")