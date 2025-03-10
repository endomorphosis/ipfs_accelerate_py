#!/usr/bin/env python3
"""
Script to validate that the updated merged_test_generator.py properly implements the template
structure needed for tests to work with the ipfs_accelerate_py worker/skillset modules.

This script:
1. Creates a new test model entry in the registry
2. Generates a test file for the model
3. Verifies the generated file has the correct structure
4. Runs the test file to ensure it executes correctly
"""

import os
import sys
import json
import shutil
import tempfile
import subprocess
import importlib.util
from pathlib import Path

# Paths
PROJECT_ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TEST_DIR = PROJECT_ROOT / "test"
WORKER_SKILLSET = PROJECT_ROOT / "ipfs_accelerate_py" / "worker" / "skillset"
GENERATOR_FILE = TEST_DIR / "merged_test_generator.py"

# Test model configuration
TEST_MODEL_NAME = "test_model"
TEST_MODEL_REGISTRY_ENTRY = """
    "test_model": {
        "family_name": "TEST_MODEL",
        "description": "Test model for validation",
        "default_model": "test-model-base",
        "class": "TestModelClass",
        "test_class": "TestTestModelClass",
        "module_name": "test_hf_test_model",
        "tasks": ["text-generation"],
        "inputs": {
            "text": "This is a test input for the model."
        },
        "dependencies": ["transformers", "tokenizers", "sentencepiece"],
        "task_specific_args": {
            "text-generation": {"max_length": 50}
        },
        "models": {
            "test-model-base": {
                "description": "Test model base version",
                "class": "TestModelClass",
                "vocab_size": 30000
            }
        }
    }
"""

def add_test_model_to_registry():
    """Add a test model to the merged_test_generator's MODEL_REGISTRY."""
    # Read the generator file
    with open(GENERATOR_FILE, 'r') as f:
        content = f.read()
    
    # Find the MODEL_REGISTRY definition
    registry_start = content.find("MODEL_REGISTRY = {")
    if registry_start == -1:
        print("Could not find MODEL_REGISTRY in merged_test_generator.py")
        return False
    
    # Find the end of the first model entry
    first_entry_end = content.find("}", registry_start)
    if first_entry_end == -1:
        print("Could not find the end of the first model entry")
        return False
    
    # Insert our test model entry
    updated_content = (
        content[:first_entry_end+1] + 
        "," + 
        TEST_MODEL_REGISTRY_ENTRY + 
        content[first_entry_end+1:]
    )
    
    # Write updated content back to file
    with open(GENERATOR_FILE, 'w') as f:
        f.write(updated_content)
    
    print(f"Added test model '{TEST_MODEL_NAME}' to MODEL_REGISTRY")
    return True

def generate_test_file():
    """Generate a test file for our test model."""
    temp_dir = Path(tempfile.mkdtemp())
    print(f"Using temporary directory: {temp_dir}")
    
    try:
        # Run the test generator
        cmd = [
            sys.executable,
            str(GENERATOR_FILE),
            "--generate", TEST_MODEL_NAME,
            "--output-dir", str(temp_dir)
        ]
        
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False
        )
        
        # Check the output
        print(f"Output: {result.stdout}")
        if result.stderr:
            print(f"Errors: {result.stderr}")
        
        # Check if the file was generated
        test_file = temp_dir / f"test_hf_{TEST_MODEL_NAME}.py"
        if not test_file.exists():
            print(f"Error: Test file not generated at {test_file}")
            return None
        
        print(f"Test file generated successfully at {test_file}")
        return test_file
    
    except Exception as e:
        print(f"Error generating test file: {e}")
        return None

def validate_test_file_structure(test_file):
    """Validate that the generated test file has the correct structure."""
    if not test_file or not test_file.exists():
        print("No test file to validate")
        return False
    
    with open(test_file, 'r') as f:
        content = f.read()
    
    # Check for key patterns that should exist in the file
    required_patterns = [
        # Class structure
        "class hf_test_model:",
        # Method definitions
        "def __init__(self, resources=None, metadata=None)",
        "self.create_cpu_text_embedding_endpoint_handler",
        "self.create_cuda_text_embedding_endpoint_handler",
        "self.create_openvino_text_embedding_endpoint_handler",
        "self.init_cpu",
        "self.init_cuda",
        "self.init_openvino",
        # Hardware platform support
        "platform: CPU",
        "platform: CUDA",
        "platform: OPENVINO"
    ]
    
    validation_results = {}
    all_patterns_found = True
    
    for pattern in required_patterns:
        found = pattern in content
        validation_results[pattern] = found
        if not found:
            all_patterns_found = False
            print(f"ERROR: Required pattern not found: '{pattern}'")
    
    # Print summary of validation results
    print("\nValidation Results:")
    for pattern, found in validation_results.items():
        status = "✅ Found" if found else "❌ Missing"
        print(f"{status}: {pattern}")
    
    return all_patterns_found

def main():
    """Main function to validate the merged_test_generator."""
    print("Starting validation of merged_test_generator.py...")
    
    # Make a backup of the generator file
    backup_file = GENERATOR_FILE.with_suffix(".py.bak_validator")
    shutil.copy2(GENERATOR_FILE, backup_file)
    print(f"Created backup of merged_test_generator.py at {backup_file}")
    
    try:
        # Add test model to registry
        if not add_test_model_to_registry():
            print("Failed to add test model to registry")
            return False
        
        # Generate test file
        test_file = generate_test_file()
        if not test_file:
            print("Failed to generate test file")
            return False
        
        # Validate test file structure
        validation_success = validate_test_file_structure(test_file)
        
        # Display the test file content
        print("\nGenerated test file content (first 100 lines):")
        with open(test_file, 'r') as f:
            lines = f.readlines()
            for i, line in enumerate(lines[:100]):
                print(f"{i+1:3d}: {line.rstrip()}")
        
        # Restore the original generator file
        shutil.copy2(backup_file, GENERATOR_FILE)
        print(f"Restored original merged_test_generator.py from backup")
        
        # Print overall result
        if validation_success:
            print("\n✅ Validation PASSED! The updated merged_test_generator.py is working correctly.")
        else:
            print("\n❌ Validation FAILED! The updated merged_test_generator.py needs further improvements.")
        
        return validation_success
    
    except Exception as e:
        print(f"Error during validation: {e}")
        
        # Try to restore the original file
        if backup_file.exists():
            shutil.copy2(backup_file, GENERATOR_FILE)
            print(f"Restored original merged_test_generator.py from backup after error")
        
        return False
    
    finally:
        # Remove the backup file
        if backup_file.exists():
            backup_file.unlink()
            print(f"Removed backup file")

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)