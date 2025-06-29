#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Model Definition Fix Script

This script updates the model definitions in execute_comprehensive_benchmarks.py
to fix issues identified by check_model_availability.py.
"""

import os
import sys
import re
from pathlib import Path

def fix_model_definitions():
    """Fix model definitions in execute_comprehensive_benchmarks.py"""
    # Path to the script
    script_path = Path(__file__).parent / 'execute_comprehensive_benchmarks.py'
    
    if not script_path.exists():
        print(f"Error: Could not find {script_path}")
        return False
    
    # Read the script content
    with open(script_path, 'r') as f:
        content = f.read()
    
    # Create backup
    backup_path = script_path.with_suffix('.py.bak')
    with open(backup_path, 'w') as f:
        f.write(content)
    print(f"Created backup at {backup_path}")
    
    # Apply fixes
    fixed_content = content
    
    # Fix for llava-next
    # Original error: llava-hf/llava-v1.6-mistral-7b is not a local folder and is not a valid model identifier listed on 'https://huggingface.co/models'
If this is a private repository, make sure to pass a token having permission to this repo either by logging in with `huggingface-cli login` or by passing `token=<your_token>`
    fixed_content = re.sub(
        r'("llava-next"\s*:\s*)"[^"]*"',
        f'\\1"llava-hf/llava-v1.6-mistral-7b"',
        fixed_content
    )
    
    # Fix for llava-next
    # Original error: llava-hf/llava-v1.6-mistral-7b is not a local folder and is not a valid model identifier listed on 'https://huggingface.co/models'
If this is a private repository, make sure to pass a token having permission to this repo either by logging in with `huggingface-cli login` or by passing `token=<your_token>`
    fixed_content = re.sub(
        r'("llava-next"\s*:\s*)"[^"]*"',
        f'\\1"llava-hf/llava-v1.6-mistral-7b"',
        fixed_content
    )
    
    # Write the fixed content
    with open(script_path, 'w') as f:
        f.write(fixed_content)
    
    print(f"Fixed model definitions in {script_path}")
    return True


if __name__ == '__main__':
    success = fix_model_definitions()
    sys.exit(0 if success else 1)