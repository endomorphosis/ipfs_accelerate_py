#!/usr/bin/env python3
"""
Script to add AMD ROCm, WebNN, and WebGPU support to model registry in test files.

This script takes an existing test file and enhances the MODEL_REGISTRY section to include
the new hardware platforms.
"""

import os
import sys
import re
import argparse
from pathlib import Path

def enhance_model_registry(test_file_path, force=False):
    """Enhance the MODEL_REGISTRY or model_info section of a test file with AMD, WebNN, and WebGPU support."""
    # Check if the file exists
    test_file = Path(test_file_path)
    if not test_file.exists():
        print(f"Error: Test file {test_file_path} does not exist")
        return False
        
    # Create a backup of the original file
    backup_file = test_file.with_suffix(test_file.suffix + '.backup')
    if not backup_file.exists() or force:
        with open(test_file, 'r') as f:
            content = f.read()
            
        with open(backup_file, 'w') as f:
            f.write(content)
            
        print(f"Created backup file: {backup_file}")
    else:
        print(f"Backup file {backup_file} already exists, skipping backup")
        with open(test_file, 'r') as f:
            content = f.read()
    
    # Check if we've already enhanced the MODEL_REGISTRY or model_info
    if 'amd' in content.lower() and 'webnn' in content.lower() and 'webgpu' in content.lower():
        print(f"Model structure in {test_file_path} already has AMD, WebNN, and WebGPU support")
        return True
    
    # Find either MODEL_REGISTRY or model_info section
    model_registry_start = content.find('MODEL_REGISTRY')
    if model_registry_start == -1:
        model_registry_start = content.find('model_info = {')
        if model_registry_start == -1:
            print(f"Error: Could not find MODEL_REGISTRY or model_info in {test_file_path}")
            return False
        print(f"Found model_info in {test_file_path}")
    else:
        print(f"Found MODEL_REGISTRY in {test_file_path}")
    
    # Add hardware compatibility if missing
    if 'hardware_compatibility' not in content and 'hardware_requirements' not in content:
        if content.find('model_info = {') != -1:
            # For model_info dictionary
            hardware_section = """        # Hardware requirements with expanded platform support
        "hardware_requirements": {
            "cpu": True,
            "cuda": True,
            "openvino": True,
            "apple": True,
            "qualcomm": False,  # Usually false for complex models
            "amd": True,  # AMD ROCm support
            "webnn": True,  # WebNN support
            "webgpu": True,   # WebGPU with transformers.js support
            "minimum_memory": "2GB",
            "recommended_memory": "4GB"
        },"""
        else:
            # For MODEL_REGISTRY
            hardware_section = """        # Hardware compatibility
        "hardware_compatibility": {
            "cpu": True,
            "cuda": True,
            "openvino": True,
            "apple": True,
            "qualcomm": False,  # Usually false for complex models
            "amd": True,  # AMD ROCm support
            "webnn": True,  # WebNN support
            "webgpu": True   # WebGPU with transformers.js support
        },"""
        
        # Find a good insertion point
        keys_to_search = [
            '"default_batch_size"', '"model_precision"', '"sequence_length"', '"embedding_dim"',
            '"model_type"', '"output_format"', '"input_format"'
        ]
        for key in keys_to_search:
            key_pos = content.find(key, model_registry_start)
            if key_pos != -1:
                line_end = content.find('\n', key_pos)
                if line_end != -1:
                    # Insert after this key
                    content = content[:line_end+1] + hardware_section + '\n' + content[line_end+1:]
                    break
    else:
        # Update existing hardware compatibility section
        if 'amd' not in content and 'webnn' not in content and 'webgpu' not in content:
            # Find the hardware_compatibility section
            hw_section_start = content.find('"hardware_compatibility"', model_registry_start)
            if hw_section_start != -1:
                opening_brace = content.find('{', hw_section_start)
                if opening_brace != -1:
                    # Find the matching closing brace
                    brace_count = 1
                    pos = opening_brace + 1
                    while brace_count > 0 and pos < len(content):
                        if content[pos] == '{':
                            brace_count += 1
                        elif content[pos] == '}':
                            brace_count -= 1
                        pos += 1
                    
                    if brace_count == 0:
                        # Found the closing brace
                        closing_brace_pos = pos - 1
                        
                        # Add the new hardware platforms before the closing brace
                        indent = '            '  # Assuming this is the indentation level
                        new_hardware = f',\n{indent}"amd": True,  # AMD ROCm support\n{indent}"webnn": True,  # WebNN support\n{indent}"webgpu": True   # WebGPU/transformers.js support'
                        content = content[:closing_brace_pos] + new_hardware + content[closing_brace_pos:]
    
    # Add precision compatibility if missing
    if 'precision_compatibility' not in content:
        if content.find('model_info = {') != -1:
            # For model_info dictionary, add to dependencies section
            precision_section = """        # Dependencies with expanded platform and precision support
        "dependencies": {
            "python": ">=3.8,<3.11",
            "pip": ["torch>=1.12.0", "transformers>=4.26.0", "numpy>=1.20.0"],
            "optional": {
                "cuda": ["nvidia-cuda-toolkit>=11.6", "nvidia-cudnn>=8.3"],
                "openvino": ["openvino>=2022.1.0"],
                "apple": ["torch>=1.12.0"],
                "qualcomm": ["qti-aisw>=1.8.0"],
                "amd": ["rocm-smi>=5.0.0", "rccl>=2.0.0", "torch-rocm>=2.0.0"],
                "webnn": ["webnn-polyfill>=1.0.0", "onnxruntime-web>=1.16.0"],
                "webgpu": ["@xenova/transformers>=2.6.0", "webgpu>=0.1.24"]
            },
            "precision": {
                "fp16": [],
                "bf16": ["torch>=1.12.0"],
                "int8": ["bitsandbytes>=0.41.0", "optimum>=1.12.0"],
                "int4": ["bitsandbytes>=0.41.0", "optimum>=1.12.0", "auto-gptq>=0.4.0"],
                "uint4": ["bitsandbytes>=0.41.0", "optimum>=1.12.0", "auto-gptq>=0.4.0"],
                "fp8": ["transformers-neuronx>=0.8.0", "torch-neuronx>=2.0.0"],
                "fp4": ["transformers-neuronx>=0.8.0", "torch-neuronx>=2.0.0"]
            }
        },"""
            
            # Find dependencies section if it exists
            deps_pos = content.find('"dependencies"', model_registry_start)
            if deps_pos != -1:
                # Replace the entire dependencies section
                opening_brace = content.find('{', deps_pos)
                if opening_brace != -1:
                    # Find the matching closing brace
                    brace_count = 1
                    pos = opening_brace + 1
                    while brace_count > 0 and pos < len(content):
                        if content[pos] == '{':
                            brace_count += 1
                        elif content[pos] == '}':
                            brace_count -= 1
                        pos += 1
                    
                    if brace_count == 0:
                        # Found the closing brace
                        closing_brace_pos = pos
                        
                        # Replace everything between the opening and closing braces
                        pre_deps = content[:deps_pos]
                        post_deps = content[closing_brace_pos:]
                        content = pre_deps + precision_section[:-1] + post_deps  # Remove trailing comma
            else:
                # Add dependencies section if it doesn't exist
                keys_to_search = [
                    '"hardware_requirements"', '"model_type"', '"output_format"', '"input_format"'
                ]
                for key in keys_to_search:
                    key_pos = content.find(key, model_registry_start)
                    if key_pos != -1:
                        line_end = content.find('\n', key_pos)
                        if line_end != -1:
                            # Insert after this key
                            content = content[:line_end+1] + precision_section + '\n' + content[line_end+1:]
                            break
        else:
            # For MODEL_REGISTRY, add precision compatibility section
            precision_section = """        # Precision support by hardware
        "precision_compatibility": {
            "cpu": {
                "fp32": True,
                "fp16": False,
                "bf16": True,
                "int8": True,
                "int4": False,
                "uint4": False,
                "fp8": False,
                "fp4": False
            },
            "cuda": {
                "fp32": True,
                "fp16": True,
                "bf16": True,
                "int8": True,
                "int4": True,
                "uint4": True,
                "fp8": False,
                "fp4": False
            },
            "openvino": {
                "fp32": True,
                "fp16": True,
                "bf16": False,
                "int8": True,
                "int4": False,
                "uint4": False,
                "fp8": False,
                "fp4": False
            },
            "apple": {
                "fp32": True,
                "fp16": True,
                "bf16": False,
                "int8": False,
                "int4": False,
                "uint4": False,
                "fp8": False,
                "fp4": False
            },
            "amd": {
                "fp32": True,
                "fp16": True,
                "bf16": True,
                "int8": True,
                "int4": False,
                "uint4": False,
                "fp8": False,
                "fp4": False
            },
            "qualcomm": {
                "fp32": True,
                "fp16": True,
                "bf16": False,
                "int8": True,
                "int4": False,
                "uint4": False,
                "fp8": False,
                "fp4": False
            },
            "webnn": {
                "fp32": True,
                "fp16": True,
                "bf16": False,
                "int8": True,
                "int4": False,
                "uint4": False,
                "fp8": False,
                "fp4": False
            },
            "webgpu": {
                "fp32": True,
                "fp16": True,
                "bf16": False,
                "int8": True,
                "int4": True,
                "uint4": False,
                "fp8": False,
                "fp4": False
            }
        },"""
        
        # Find the hardware_compatibility section we just added or updated
        hw_section_start = content.find('"hardware_compatibility"', model_registry_start)
        if hw_section_start != -1:
            # Find the end of the hardware_compatibility section
            opening_brace = content.find('{', hw_section_start)
            if opening_brace != -1:
                # Find the matching closing brace
                brace_count = 1
                pos = opening_brace + 1
                while brace_count > 0 and pos < len(content):
                    if content[pos] == '{':
                        brace_count += 1
                    elif content[pos] == '}':
                        brace_count -= 1
                    pos += 1
                
                if brace_count == 0:
                    # Found the closing brace
                    closing_brace_pos = pos - 1
                    
                    # Find the end of the line containing the closing brace
                    line_end = content.find('\n', closing_brace_pos)
                    if line_end != -1:
                        # Insert after the hardware_compatibility section
                        content = content[:line_end+1] + precision_section + '\n' + content[line_end+1:]
    
    # Write the updated content back to the file
    with open(test_file, 'w') as f:
        f.write(content)
        
    print(f"Enhanced MODEL_REGISTRY in {test_file_path}")
    return True

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Add AMD, WebNN, and WebGPU support to MODEL_REGISTRY in test files")
    parser.add_argument("--test-file", type=str, required=True, help="Path to the test file to enhance")
    parser.add_argument("--force", action="store_true", help="Force overwrite of backup file if it exists")
    
    args = parser.parse_args()
    
    # Enhance the test file
    success = enhance_model_registry(args.test_file, args.force)
    
    if success:
        print(f"Successfully enhanced MODEL_REGISTRY in {args.test_file}")
        return 0
    else:
        print(f"Failed to enhance MODEL_REGISTRY in {args.test_file}")
        return 1

if __name__ == "__main__":
    sys.exit(main())