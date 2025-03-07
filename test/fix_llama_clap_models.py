#\!/usr/bin/env python3
"""
Fix LLAMA and CLAP model registration in run_model_benchmarks.py

This script adds or fixes the LLAMA and CLAP model keys in the run_model_benchmarks.py
script to enable proper benchmarking for these models.
"""

import os
import re
import sys
from pathlib import Path

def fix_model_registration():
    """Fix model registration for LLAMA and CLAP in run_model_benchmarks.py."""
    # File to modify
    benchmark_file = Path(__file__).parent / 'run_model_benchmarks.py'
    
    if not benchmark_file.exists():
        print(f"Error: Could not find {benchmark_file}")
        return False
    
    # Create backup
    backup_file = benchmark_file.with_suffix('.py.bak')
    with open(benchmark_file, 'r') as f:
        original_content = f.read()
    
    with open(backup_file, 'w') as f:
        f.write(original_content)
    
    print(f"Created backup at {backup_file}")
    
    # Check if LLAMA and CLAP are already in KEY_MODEL_SET or SMALL_MODEL_SET
    llama_in_key = 'llama' in original_content and 'TinyLlama/TinyLlama-1.1B-Chat-v1.0' in original_content
    clap_in_key = 'clap' in original_content and 'laion/clap-htsat-unfused' in original_content
    
    if llama_in_key and clap_in_key:
        # Find locations of model sets
        key_model_set_pattern = r'KEY_MODEL_SET\s*=\s*\{[^}]*\}'
        small_model_set_pattern = r'SMALL_MODEL_SET\s*=\s*\{[^}]*\}'
        
        key_model_match = re.search(key_model_set_pattern, original_content, re.DOTALL)
        small_model_match = re.search(small_model_set_pattern, original_content, re.DOTALL)
        
        if not key_model_match or not small_model_match:
            print("Error: Could not locate KEY_MODEL_SET or SMALL_MODEL_SET in the file.")
            return False
        
        # Check if keys are directly registered
        if 'llama' not in key_model_match.group(0) or 'clap' not in key_model_match.group(0):
            print("Error: Keys exist but not in the expected format. Manual inspection needed.")
            return False
        
        print("LLAMA and CLAP models are already registered correctly.")
        print("The issue might be in how the models are accessed. Let's check specific_models handling.")
        
        # Fix the specific_models handling to ensure it properly matches keys
        content = original_content
        model_key_pattern = r'(model_set\s*=\s*\{[^}]*\})'
        model_set_code = re.search(model_key_pattern, content, re.DOTALL)
        
        if model_set_code:
            # Add more debugging
            debug_code = """
            # Add debugging for model key issues
            if 'llama' in args.specific_models or 'clap' in args.specific_models:
                print(f"Debugging model keys: {args.specific_models}")
                print(f"Available keys: {sorted(model_set.keys())}")
            """
            
            # Place after model_set code
            new_content = content.replace(
                model_set_code.group(0),
                f"{model_set_code.group(0)}\n{debug_code}"
            )
            
            # Write the updated content
            with open(benchmark_file, 'w') as f:
                f.write(new_content)
            
            print("Added debugging code for model key matching.")
            return True
        else:
            print("Could not locate model_set code. Manual inspection needed.")
            return False
    
    # If we got here, we need to update the model registrations
    # Find KEY_MODEL_SET and update it
    key_model_set_pattern = r'(KEY_MODEL_SET\s*=\s*\{[^}]*)(})'
    content = re.sub(
        key_model_set_pattern,
        r'\1,\n    "llama": {"name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "family": "text_generation", "size": "small", "modality": "text"},\n    "clap": {"name": "laion/clap-htsat-unfused", "family": "audio", "size": "base", "modality": "audio"}\2',
        original_content, 
        flags=re.DOTALL
    )
    
    # Find SMALL_MODEL_SET and update it
    small_model_set_pattern = r'(SMALL_MODEL_SET\s*=\s*\{[^}]*)(})'
    content = re.sub(
        small_model_set_pattern,
        r'\1,\n    "llama": {"name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "family": "text_generation", "size": "small", "modality": "text"},\n    "clap": {"name": "laion/clap-htsat-unfused", "family": "audio", "size": "base", "modality": "audio"}\2',
        content, 
        flags=re.DOTALL
    )
    
    # Write the updated content
    with open(benchmark_file, 'w') as f:
        f.write(content)
    
    print("Updated model registrations for LLAMA and CLAP in run_model_benchmarks.py")
    return True

if __name__ == "__main__":
    success = fix_model_registration()
    sys.exit(0 if success else 1)
