#!/usr/bin/env python3
"""
Update the merged_test_generator.py file to incorporate hardware-aware templates.

This script:
1. Updates the merged_test_generator.py to use hardware-aware templates
2. Adds hardware platform selection capabilities
3. Integrates generated templates from hardware_test_templates directory
4. Makes the generator capable of creating tests with support for all hardware platforms

Usage:
  python update_test_generator_with_hardware_templates.py
"""

import os
import re
import sys
import shutil
import json
from pathlib import Path
from datetime import datetime

# Paths
PROJECT_ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TEST_DIR = PROJECT_ROOT / "test"
GENERATOR_FILE = TEST_DIR / "merged_test_generator.py"
TEMPLATE_DIR = TEST_DIR / "hardware_test_templates"
TEMPLATE_DB_FILE = TEMPLATE_DIR / "template_database.json"

# Ensure template database exists
if not TEMPLATE_DB_FILE.exists():
    print(f"Error: Template database not found at {TEMPLATE_DB_FILE}")
    print("Run enhance_key_models_hardware_coverage.py --update-generator first")
    sys.exit(1)

# Create a backup of the generator file
def backup_generator():
    """Create a backup of the generator file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_file = GENERATOR_FILE.with_suffix(f".py.bak_{timestamp}")
    shutil.copy2(GENERATOR_FILE, backup_file)
    print(f"Created backup of {GENERATOR_FILE} at {backup_file}")
    return backup_file

# Load template database
def load_template_database():
    """Load the template database."""
    with open(TEMPLATE_DB_FILE, 'r') as f:
        template_db = json.load(f)
    return template_db

# Update generator file with hardware-aware templates
def update_generator(template_db):
    """
    Update the generator file with hardware-aware templates.
    
    Args:
        template_db: Template database
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Read the generator file
        with open(GENERATOR_FILE, 'r') as f:
            content = f.read()
        
        # Add hardware platform options
        if "--platform" not in content:
            # Find the argument parser section
            parser_section = re.search(r'parser = argparse\.ArgumentParser\(.*?\)', content, re.DOTALL)
            if parser_section:
                parser_pos = content.find(parser_section.group(0)) + len(parser_section.group(0))
                
                # Add platform argument
                platform_arg = """
    # Hardware platform options
    parser.add_argument("--platform", choices=["cpu", "cuda", "openvino", "mps", "rocm", "webnn", "webgpu", "all"],
                      default="all", help="Hardware platform to target (default: all)")
"""
                content = content[:parser_pos] + platform_arg + content[parser_pos:]
        
        # Add hardware-aware template selection functionality
        if "select_hardware_template" not in content:
            # Find a good place to add the function - before the main generate_test_file function
            generate_func_match = re.search(r'def generate_test_file\(', content)
            if generate_func_match:
                insert_pos = content.rfind("\n\n", 0, generate_func_match.start())
                
                # Add template selection function
                template_func = """
def select_hardware_template(model_name, category=None, platform="all"):
    """
    Select an appropriate template based on model name, category, and target hardware platform.
    
    Args:
        model_name: Name of the model
        category: Category of the model (text, vision, audio, etc.)
        platform: Target hardware platform (cpu, cuda, openvino, mps, rocm, webnn, webgpu, all)
        
    Returns:
        str: Template content
    """
    # Try model-specific template first
    template_key = model_name
    if template_key in template_database:
        return template_database[template_key]
    
    # Try category template next
    if category and category in template_database:
        return template_database[category]
    
    # Try key models mapping for 13 high-priority models
    key_models_mapping = {
        "bert": "text_embedding", 
        "gpt2": "text_generation",
        "t5": "text_generation",
        "llama": "text_generation",
        "vit": "vision",
        "clip": "vision",
        "whisper": "audio",
        "wav2vec2": "audio",
        "clap": "audio",
        "detr": "vision",
        "llava": "vision_language",
        "llava_next": "vision_language",
        "qwen2": "text_generation",
        "xclip": "video"
    }
    
    # Check if this is a known key model type
    for key_prefix, mapped_category in key_models_mapping.items():
        if model_name.lower().startswith(key_prefix.lower()):
            if mapped_category in template_database:
                print(f"Using {mapped_category} template for {model_name}")
                return template_database[mapped_category]
            elif key_prefix in template_database:
                print(f"Using {key_prefix} template for {model_name}")
                return template_database[key_prefix]
    
    # Default to generic template
    if "generic" in template_database:
        return template_database["generic"]
    
    # Fall back to built-in template if no matches
    return DEFAULT_TEMPLATE
"""
                content = content[:insert_pos] + template_func + content[insert_pos:]
        
        # Add template database initialization code
        if "template_database =" not in content:
            # Find a good place to add the database - near the top of the file
            imports_end = content.find("\n\n", content.find("import"))
            if imports_end == -1:
                imports_end = content.find("import") + 500  # Rough estimate
            
            # Add template database initialization
            template_db_code = """
# Hardware-aware templates
template_database = {}
"""
            
            # Include key templates in the code
            for key, template in template_db.items():
                if key in ["text_embedding", "text_generation", "vision", "audio", "vision_language", "video"]:
                    template_db_code += f'\ntemplate_database["{key}"] = """'
                    template_db_code += template[:1000]  # Truncate for readability
                    template_db_code += '..."""  # Truncated for readability'
            
            content = content[:imports_end] + template_db_code + content[imports_end:]
        
        # Update the generate_test_file function to use hardware-aware templates
        if "select_hardware_template" not in content or "platform=" not in content:
            generate_func = re.search(r'def generate_test_file\([^)]*\):', content)
            if generate_func:
                # Update function signature
                old_signature = generate_func.group(0)
                new_signature = old_signature.replace("):", ", platform=\"all\"):")
                content = content.replace(old_signature, new_signature)
                
                # Update template selection code
                template_selection = re.search(r'template = .*?DEFAULT_TEMPLATE', content, re.DOTALL)
                if template_selection:
                    old_template_code = template_selection.group(0)
                    new_template_code = f"template = select_hardware_template(model_name, category, platform)"
                    content = content.replace(old_template_code, new_template_code)
        
        # Update the main function to pass platform parameter
        main_func = re.search(r'def main\(\):', content)
        if main_func:
            # Find the generate_test_file call
            generate_call = re.search(r'generate_test_file\([^)]*\)', content)
            if generate_call:
                old_call = generate_call.group(0)
                if "platform=" not in old_call:
                    new_call = old_call.replace(")", ", platform=args.platform)")
                    content = content.replace(old_call, new_call)
        
        # Write the updated content
        with open(GENERATOR_FILE, 'w') as f:
            f.write(content)
        
        print(f"Successfully updated {GENERATOR_FILE} with hardware-aware templates")
        return True
    
    except Exception as e:
        print(f"Error updating generator file: {e}")
        return False

def main():
    """Main function."""
    # Back up the generator file
    backup_file = backup_generator()
    
    # Load template database
    template_db = load_template_database()
    
    # Update generator
    success = update_generator(template_db)
    
    if success:
        print("Generator update completed successfully!")
    else:
        print(f"Generator update failed. Original file restored from {backup_file}")
        shutil.copy2(backup_file, GENERATOR_FILE)

if __name__ == "__main__":
    main()