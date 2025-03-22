#!/usr/bin/env python3
"""
Script to update enhanced_generator.py with additional models
from additional_models.py to achieve 100% coverage of the models
tracked in HF_MODEL_COVERAGE_ROADMAP.md.
"""

import os
import re
import sys
from additional_models import ADDITIONAL_MODELS, ADDITIONAL_ARCHITECTURE_MAPPINGS

def update_generator_file(generator_file, backup=True):
    """Update the generator file with additional models."""
    # Read the original file
    with open(generator_file, 'r') as f:
        content = f.read()
    
    if backup:
        # Create a backup
        backup_file = f"{generator_file}.bak"
        with open(backup_file, 'w') as f:
            f.write(content)
        print(f"Created backup at {backup_file}")
    
    # Find the MODEL_REGISTRY closing bracket
    model_registry_end = re.search(r'}\s*\n\s*def\s+get_model_architecture', content, re.DOTALL)
    if not model_registry_end:
        print("Error: Could not find the end of MODEL_REGISTRY in the file!")
        return False
    
    # Find the ARCHITECTURE_TYPES dict
    arch_types_match = re.search(r'ARCHITECTURE_TYPES\s*=\s*{([^}]*)}', content, re.DOTALL)
    if not arch_types_match:
        print("Error: Could not find ARCHITECTURE_TYPES in the file!")
        return False
    
    # Build the additional model entries
    additional_models_str = ""
    for model_name, model_config in ADDITIONAL_MODELS.items():
        # Skip if the model is already in MODEL_REGISTRY
        if f'"{model_name}"' in content or f"'{model_name}'" in content:
            print(f"Model {model_name} already exists in MODEL_REGISTRY, skipping...")
            continue
        
        additional_models_str += f"""
    "{model_name}": {{
        "default_model": "{model_config['default_model']}",
        "task": "{model_config['task']}",
        "class": "{model_config['class']}",
        "test_input": {repr(model_config['test_input'])}
    }},"""
    
    # Insert the additional models before the closing bracket
    insert_pos = model_registry_end.start()
    new_content = content[:insert_pos] + additional_models_str + content[insert_pos:]
    
    # Now update ARCHITECTURE_TYPES
    arch_types_content = arch_types_match.group(1)
    new_arch_types_content = arch_types_content
    
    for arch_type, models in ADDITIONAL_ARCHITECTURE_MAPPINGS.items():
        # Find the architecture type in the dict
        arch_match = re.search(rf'"{arch_type}":\s*\[(.*?)\]', new_arch_types_content, re.DOTALL)
        if arch_match:
            # Get the current models for this architecture
            current_models = arch_match.group(1)
            # Add the new models
            new_models = current_models
            for model in models:
                # Only add if not already in the list
                if f'"{model}"' not in current_models and f"'{model}'" not in current_models:
                    if new_models.strip().endswith(','):
                        new_models += f' "{model}",'
                    else:
                        new_models += f', "{model}"'
            
            # Replace in the architecture types content
            new_arch_types_content = new_arch_types_content.replace(current_models, new_models)
    
    # Replace the ARCHITECTURE_TYPES in the content
    new_content = new_content.replace(arch_types_match.group(1), new_arch_types_content)
    
    # Write the updated content
    with open(generator_file, 'w') as f:
        f.write(new_content)
    
    print(f"Updated {generator_file} with additional models!")
    return True

def main():
    """Main function to update the generator file."""
    if len(sys.argv) > 1:
        generator_file = sys.argv[1]
    else:
        # Default location
        generator_file = 'enhanced_generator.py'
    
    # Check if file exists
    if not os.path.isfile(generator_file):
        print(f"Error: Generator file '{generator_file}' not found!")
        return 1
    
    success = update_generator_file(generator_file)
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())