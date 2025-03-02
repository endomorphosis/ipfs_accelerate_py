#!/usr/bin/env python3
"""
Script to update the merged_test_generator.py file to better align with
the implementation structure of the ipfs_accelerate_py worker/skillset modules.

This script will:
1. Update the generator template to match the worker/skillset implementation structure
2. Add support for handler creation methods that follow the implementation pattern
3. Enhance the mock interface to match real implementation classes
"""

import os
import sys
import json
import glob
import shutil
import datetime
from pathlib import Path

# Determine the project root and relevant directories
PROJECT_ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TEST_DIR = PROJECT_ROOT / "test"
WORKER_SKILLSET = PROJECT_ROOT / "ipfs_accelerate_py" / "worker" / "skillset"
GENERATOR_FILE = TEST_DIR / "merged_test_generator.py"

def backup_existing_generator():
    """Create a backup of the existing merged_test_generator.py file."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = GENERATOR_FILE.with_suffix(f".py.bak_{timestamp}")
    
    print(f"Creating backup of merged_test_generator.py at {backup_path}")
    shutil.copy2(GENERATOR_FILE, backup_path)
    return backup_path

def get_all_hf_skillset_modules():
    """Get all HF model implementations from the worker/skillset directory."""
    hf_modules = []
    
    for file_path in WORKER_SKILLSET.glob("hf_*.py"):
        module_name = file_path.stem  # Get file name without extension
        if "test" not in module_name:  # Skip test files
            hf_modules.append({
                "name": module_name,
                "path": file_path,
                "model_type": module_name[3:] if module_name.startswith("hf_") else module_name
            })
    
    print(f"Found {len(hf_modules)} HF model implementations in worker/skillset")
    return sorted(hf_modules, key=lambda x: x["name"])

def extract_handler_methods(module_file):
    """Extract handler creation method patterns from an implementation file."""
    handler_methods = []
    
    with open(module_file, 'r') as f:
        content = f.read()
        
    # Look for handler creation methods
    method_markers = [
        "def create_cpu_",
        "def create_cuda_",
        "def create_openvino_",
        "def create_apple_",
        "def create_qualcomm_"
    ]
    
    for marker in method_markers:
        if marker in content:
            print(f"Found handler method pattern: {marker}")
            handler_methods.append(marker)
    
    return handler_methods

def update_template_structure():
    """Update the test file template in merged_test_generator.py to match worker/skillset structure."""
    
    # Get a model module as reference
    hf_modules = get_all_hf_skillset_modules()
    if not hf_modules:
        print("No HF model modules found in worker/skillset")
        return False
    
    # Backup the existing file first
    backup_path = backup_existing_generator()
    
    # Create reference model set from different categories
    reference_models = {
        "language": next((m for m in hf_modules if m["model_type"] in ["bert", "t5", "llama"]), None),
        "vision": next((m for m in hf_modules if m["model_type"] in ["clip", "vit", "detr"]), None),
        "audio": next((m for m in hf_modules if m["model_type"] in ["whisper", "wav2vec2", "clap"]), None),
        "multimodal": next((m for m in hf_modules if m["model_type"] in ["llava", "llava_next"]), None)
    }
    
    # Extract handler methods from reference models
    handler_patterns = {}
    for category, module in reference_models.items():
        if module:
            print(f"Analyzing {category} reference model: {module['name']}")
            handler_patterns[category] = extract_handler_methods(module['path'])
    
    # Read the existing generator file
    with open(GENERATOR_FILE, 'r') as f:
        generator_content = f.read()
    
    # Update the template in the generator file
    # This is where we'll modify the template to align with worker/skillset structure
    
    # 1. Update the class structure to match worker/skillset
    print("Updating class structure to match worker/skillset pattern...")
    
    # Parse out the template definition section
    template_start = generator_content.find("def generate_test_template(")
    if template_start == -1:
        print("Could not find template generation function")
        return False
    
    template_end = generator_content.find("def extract_implementation_status(", template_start)
    if template_end == -1:
        template_end = len(generator_content)
    
    template_section = generator_content[template_start:template_end]
    
    # Key patterns to update
    updates = [
        # Update class structure initialization
        (
            "class {test_class_name}:",
            "class {class_name}:\n    \"\"\"{model} implementation.\n    \n    This class provides standardized interfaces for working with {model} models\n    across different hardware backends (CPU, CUDA, OpenVINO, Apple, Qualcomm).\n    \"\"\""
        ),
        # Update initialization to match worker/skillset
        (
            "def __init__(self, resources=None, metadata=None):",
            "def __init__(self, resources=None, metadata=None):\n        \"\"\"Initialize the {model} model.\n        \n        Args:\n            resources (dict): Dictionary of shared resources (torch, transformers, etc.)\n            metadata (dict): Configuration metadata\n        \"\"\""
        ),
        # Add handler creation methods
        (
            "# Initialize model handler",
            "# Handler creation methods\n        self.create_cpu_text_embedding_endpoint_handler = self.create_cpu_text_embedding_endpoint_handler\n        self.create_cuda_text_embedding_endpoint_handler = self.create_cuda_text_embedding_endpoint_handler\n        self.create_openvino_text_embedding_endpoint_handler = self.create_openvino_text_embedding_endpoint_handler\n        self.create_apple_text_embedding_endpoint_handler = self.create_apple_text_embedding_endpoint_handler\n        self.create_qualcomm_text_embedding_endpoint_handler = self.create_qualcomm_text_embedding_endpoint_handler\n        \n        # Initialization methods\n        self.init = self.init\n        self.init_cpu = self.init_cpu\n        self.init_cuda = self.init_cuda\n        self.init_openvino = self.init_openvino\n        self.init_apple = self.init_apple\n        self.init_qualcomm = self.init_qualcomm\n        \n        # Test methods\n        self.__test__ = self.__test__"
        ),
    ]
    
    # Apply the updates
    updated_template_section = template_section
    for old_pattern, new_pattern in updates:
        updated_template_section = updated_template_section.replace(old_pattern, new_pattern)
    
    # Combine everything back together
    updated_generator_content = (
        generator_content[:template_start] +
        updated_template_section +
        generator_content[template_end:]
    )
    
    # Write the updated content back to the file
    with open(GENERATOR_FILE, 'w') as f:
        f.write(updated_generator_content)
    
    print(f"Updated merged_test_generator.py with worker/skillset structure")
    print(f"Original file backed up at: {backup_path}")
    
    return True

if __name__ == "__main__":
    print("Starting update of merged_test_generator.py to align with worker/skillset implementation...")
    success = update_template_structure()
    
    if success:
        print("\nUpdate completed successfully!")
        print("\nRecommended next steps:")
        print("1. Review the changes made to merged_test_generator.py")
        print("2. Test the updated generator by running: python merged_test_generator.py --generate bert")
        print("3. Verify the generated test file matches the worker/skillset structure")
    else:
        print("\nUpdate failed! Check the error messages above.")