#!/usr/bin/env python3
"""
Debug script for testing the multimodal template.
This script allows quick validation of the multimodal template by generating a test file
for a specified model and printing it to the console.
"""

import os
import sys
import argparse
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import template utilities
from template_integration.template_integration_workflow import fix_indentation
from template_integration.generate_refactored_test import sanitize_model_name

def debug_template(model_name, architecture="multimodal"):
    """Debug the multimodal template with a specific model."""
    print(f"Debugging {architecture} template with model: {model_name}")
    
    # Get template path
    template_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "templates",
        f"refactored_{architecture}_template.py"
    )
    
    # Check if template exists
    if not os.path.exists(template_path):
        print(f"Error: Template not found at {template_path}")
        return False
    
    # Read template content
    with open(template_path, 'r') as f:
        template_content = f.read()
    
    # Get sanitized model name
    sanitized_model_name = sanitize_model_name(model_name)
    
    # Replace placeholders
    customized_content = template_content.replace("{model_name}", model_name)
    customized_content = customized_content.replace("{sanitized_model_name}", sanitized_model_name)
    customized_content = customized_content.replace("{timestamp}", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    customized_content = customized_content.replace("{architecture}", architecture)
    customized_content = customized_content.replace("{base_class}", "ModelTest")
    
    # Fix indentation issues
    fixed_content = fix_indentation(customized_content, model_name, architecture)
    
    # Print the result
    print("\n" + "="*80)
    print(f"Generated test file for {model_name} ({architecture}):")
    print("="*80)
    print(fixed_content)
    print("="*80)
    
    # Optionally save to file
    output_path = f"debug_{sanitized_model_name}_{architecture}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.py"
    with open(output_path, 'w') as f:
        f.write(fixed_content)
    print(f"Saved debug output to {output_path}")
    
    return True

def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Debug multimodal template generation")
    parser.add_argument("--model", type=str, default="openai/clip-vit-base-patch32", 
                        help="Model name (default: openai/clip-vit-base-patch32)")
    args = parser.parse_args()
    
    # Run template debugging
    success = debug_template(args.model)
    
    # Return appropriate exit code
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())