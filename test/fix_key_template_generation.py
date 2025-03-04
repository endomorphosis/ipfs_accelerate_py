#!/usr/bin/env python3
"""
Quick fix for the merged_test_generator.py to ensure hardware-aware template generation.
"""

import os
import shutil
import sys
from pathlib import Path
import re

# Basic modifications to ensure hardware support in the generators
def fix_generator():
    """Fix the generator files to support hardware platforms."""
    # Path to files
    generator_file = "/home/barberb/ipfs_accelerate_py/test/merged_test_generator.py"
    
    # Check if file exists
    if not os.path.exists(generator_file):
        print(f"Error: File not found: {generator_file}")
        return False
    
    # Create backup
    backup_file = f"{generator_file}.bak"
    shutil.copy2(generator_file, backup_file)
    print(f"Created backup at {backup_file}")
    
    # Read the file
    with open(generator_file, 'r') as f:
        content = f.read()
    
    # Add key hardware-related constants
    hardware_constants = """
# Hardware platforms to support
HARDWARE_PLATFORMS = ["cpu", "cuda", "openvino", "mps", "rocm", "webnn", "webgpu"]

# Model categories
MODEL_CATEGORIES = {
    "bert": "text_embedding",
    "clap": "audio",
    "clip": "vision",
    "detr": "vision",
    "llama": "text_generation",
    "llava": "vision_language",
    "llava_next": "vision_language",
    "qwen2": "text_generation",
    "t5": "text_generation",
    "vit": "vision",
    "wav2vec2": "audio",
    "whisper": "audio",
    "xclip": "video"
}

# Hardware-specific modifiers for detection
HARDWARE_DETECTION = {
    "cuda": "if torch.cuda.is_available()",
    "mps": "if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()",
    "rocm": "if torch.cuda.is_available() and hasattr(torch.version, 'hip')",
    "openvino": "try:\\n    import openvino\\n    OPENVINO_AVAILABLE = True\\nexcept ImportError:\\n    OPENVINO_AVAILABLE = False\\n",
    "webnn": "# WebNN detection would depend on browser environment",
    "webgpu": "# WebGPU detection would depend on browser environment"
}
"""
    
    # Add constants if not present
    if "HARDWARE_PLATFORMS =" not in content:
        # Find import section
        import_pos = content.find("from typing import")
        if import_pos > 0:
            import_end = content.find("\n", import_pos)
            content = content[:import_end+1] + hardware_constants + content[import_end+1:]
    
    # Add platform parameter to generate_test_file function
    generate_func_match = content.find("def generate_test_file(")
    if generate_func_match > 0 and "platform=" not in content[generate_func_match:generate_func_match+500]:
        # Update function signature
        old_sig = content[generate_func_match:content.find(")", generate_func_match)+1]
        new_sig = old_sig.replace(")", ", platform=\"all\")")
        content = content.replace(old_sig, new_sig)
    
    # Add platform parameter to CLI arguments
    parser_match = content.find("parser = argparse.ArgumentParser(")
    if parser_match > 0 and "--platform" not in content[parser_match:parser_match+1000]:
        # Find end of argument definitions
        args_end = content.find("return parser.parse_args()", parser_match)
        if args_end > 0:
            # Add platform argument
            platform_arg = """
    # Hardware platform options
    parser.add_argument("--platform", choices=["cpu", "cuda", "openvino", "mps", "rocm", "webnn", "webgpu", "all"],
                      default="all", help="Hardware platform to target (default: all)")
"""
            content = content[:args_end] + platform_arg + content[args_end:]
    
    # Add platform parameter to template generation
    template_gen_match = content.find("template = generate_test_template(")
    if template_gen_match > 0:
        # Update the template call to include platform parameter
        old_call = content[template_gen_match:content.find(")", template_gen_match)+1]
        if "platform=" not in old_call:
            new_call = old_call.replace(")", ", platform=platform)")
            content = content.replace(old_call, new_call)
    
    # Update main function to pass platform parameter
    main_func_match = content.find("def main():")
    if main_func_match > 0:
        generate_call_match = content.find("generate_test_file(", main_func_match)
        while generate_call_match > 0:
            # Find the call
            old_call = content[generate_call_match:content.find(")", generate_call_match)+1]
            if "platform=" not in old_call:
                # Update the call
                new_call = old_call.replace(")", ", platform=args.platform)")
                content = content.replace(old_call, new_call)
            generate_call_match = content.find("generate_test_file(", generate_call_match+1)
    
    # Write back the updated content
    with open(generator_file, 'w') as f:
        f.write(content)
    
    print(f"Successfully updated {generator_file} with hardware platform support")
    return True

if __name__ == "__main__":
    fix_generator()