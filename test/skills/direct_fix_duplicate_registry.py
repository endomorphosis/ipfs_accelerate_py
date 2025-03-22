#!/usr/bin/env python3

"""
Directly fix the MODEL_REGISTRY duplication issue in test_generator_fixed.py
"""

import os
import sys
import time
import re
import json
import shutil

def direct_fix_registry(file_path):
    """Directly fix the MODEL_REGISTRY duplication by reconstructing it."""
    print(f"Fixing {file_path}...")
    
    # Create backup
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    backup_path = f"{file_path}.bak.{timestamp}"
    shutil.copy2(file_path, backup_path)
    print(f"Created backup at {backup_path}")
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Load the model registry data from huggingface_model_types.json
    registry_file = os.path.join(os.path.dirname(file_path), 'huggingface_model_types.json')
    if os.path.exists(registry_file):
        try:
            with open(registry_file, 'r') as f:
                registry_data = json.load(f)
            print(f"Loaded registry data from {registry_file}")
        except Exception as e:
            print(f"Error loading registry file: {e}")
            registry_data = {}
    else:
        print("Registry file not found, no updates will be applied")
        registry_data = {}
    
    # Find MODEL_REGISTRY section beginning
    registry_start = content.find("MODEL_REGISTRY = {")
    if registry_start == -1:
        print("Error: MODEL_REGISTRY not found")
        return False
        
    # Find the CLASS_NAME_FIXES section which comes after MODEL_REGISTRY
    fixes_start = content.find("# Class name capitalization fixes", registry_start)
    if fixes_start == -1:
        fixes_start = content.find("CLASS_NAME_FIXES = {", registry_start)
    
    if fixes_start == -1:
        print("Error: Could not find section after MODEL_REGISTRY")
        return False
    
    # Now extract all model entries
    # Match pattern for model entries: "model_type": { ... }, 
    model_pattern = re.compile(r'"([\w-]+)":\s*{(.+?)},\s*(?="|\n\})', re.DOTALL)
    matches = list(model_pattern.finditer(content, registry_start, fixes_start))
    
    if not matches:
        print("Error: Could not find model entries in MODEL_REGISTRY")
        return False
    
    # Build cleaned MODEL_REGISTRY
    models = {}
    for match in matches:
        model_type = match.group(1)
        model_config = match.group(2).strip()
        
        # Only keep the first instance of each model type
        if model_type not in models:
            models[model_type] = model_config
    
    print(f"Found {len(matches)} model entries, keeping {len(models)} unique models")
    
    # Build the new MODEL_REGISTRY section
    new_registry = "MODEL_REGISTRY = {\n"
    for model_type, config in models.items():
        # Update with registry data if available
        if model_type in registry_data and 'default_model' in registry_data[model_type]:
            # Find the default_model line in the config
            default_pattern = re.compile(r'"default_model":\s*"[^"]*"')
            new_default = f'"default_model": "{registry_data[model_type]["default_model"]}"'
            if default_pattern.search(config):
                config = default_pattern.sub(new_default, config)
                print(f"Updated default model for {model_type} to {registry_data[model_type]['default_model']}")
        
        new_registry += f'    "{model_type}": {{\n        {config}\n    }},\n'
    new_registry += "}\n\n"
    
    # Replace the old MODEL_REGISTRY with the new one
    new_content = content[:registry_start] + new_registry + content[fixes_start:]
    
    # Write back to file
    with open(file_path, 'w') as f:
        f.write(new_content)
    
    # Verify the file
    try:
        compile(new_content, file_path, 'exec')
        print("✅ Fixed file is syntactically valid")
        return True
    except SyntaxError as e:
        print(f"❌ Syntax error still exists at line {e.lineno}: {e.msg}")
        print("Failed to fix file. Restoring backup.")
        shutil.copy2(backup_path, file_path)
        return False

def main():
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        file_path = "test_generator_fixed.py"
    
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found")
        return 1
    
    success = direct_fix_registry(file_path)
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())