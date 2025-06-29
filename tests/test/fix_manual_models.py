#!/usr/bin/env python3
"""
Fix manually created model test files by regenerating them with the template system.

This script:
1. Identifies manually created model tests that don't follow template structure
2. Maps each model to its correct architecture type
3. Regenerates the tests using the appropriate template
4. Updates the architecture types and model registry
5. Ensures proper syntax and structure conformance

Usage:
    python fix_manual_models.py [--verify] [--apply]
"""

import os
import sys
import argparse
import logging
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"fix_manual_models_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    ]
)
logger = logging.getLogger(__name__)

# Define paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SKILLS_DIR = os.path.join(SCRIPT_DIR, "skills")
TEMPLATES_DIR = os.path.join(SKILLS_DIR, "templates")
FINAL_MODELS_DIR = os.path.join(SCRIPT_DIR, "final_models")
FIXED_TESTS_DIR = os.path.join(SKILLS_DIR, "fixed_tests")

# Ensure fixed_tests directory exists
os.makedirs(FIXED_TESTS_DIR, exist_ok=True)

# Define manual models and their architecture mappings
MANUAL_MODELS = {
    "layoutlmv2": {
        "architecture": "vision-encoder-text-decoder",
        "template": "vision_text_template.py",
        "model_id": "microsoft/layoutlmv2-base-uncased",
        "class": "LayoutLMv2ForSequenceClassification",
        "source_file": os.path.join(FINAL_MODELS_DIR, "test_layoutlmv2.py")
    },
    "layoutlmv3": {
        "architecture": "vision-encoder-text-decoder",
        "template": "vision_text_template.py",
        "model_id": "microsoft/layoutlmv3-base",
        "class": "LayoutLMv3ForSequenceClassification",
        "source_file": os.path.join(FINAL_MODELS_DIR, "test_layoutlmv3.py")
    },
    "clvp": {
        "architecture": "speech",
        "template": "speech_template.py",
        "model_id": "susnato/clvp_dev",
        "class": "CLVPForCausalLM",
        "source_file": os.path.join(FINAL_MODELS_DIR, "test_clvp.py")
    },
    "bigbird": {
        "architecture": "encoder-decoder",
        "template": "encoder_decoder_template.py",
        "model_id": "google/bigbird-roberta-base",
        "class": "BigBirdForSequenceClassification",
        "source_file": os.path.join(FINAL_MODELS_DIR, "test_hf_bigbird.py")
    },
    "seamless_m4t_v2": {
        "architecture": "speech",
        "template": "speech_template.py",
        "model_id": "facebook/seamless-m4t-v2-large",
        "class": "SeamlessM4TModel",
        "source_file": os.path.join(FINAL_MODELS_DIR, "test_seamless_m4t_v2.py")
    },
    "xlm_prophetnet": {
        "architecture": "encoder-decoder",
        "template": "encoder_decoder_template.py",
        "model_id": "microsoft/xprophetnet-large-wiki100-cased",
        "class": "XLMProphetNetForConditionalGeneration",
        "source_file": os.path.join(FINAL_MODELS_DIR, "test_xlm_prophetnet.py")
    }
}

def get_template_content(template_name):
    """Get the content of a template file."""
    template_path = os.path.join(TEMPLATES_DIR, template_name)
    try:
        with open(template_path, 'r') as f:
            return f.read()
    except FileNotFoundError:
        logger.error(f"Template not found: {template_path}")
        return None

def generate_test_from_template(model_name, model_info, output_dir):
    """Generate a test file from a template."""
    template_content = get_template_content(model_info["template"])
    if not template_content:
        return False
    
    # Create a backup of the original file if it exists
    target_path = os.path.join(output_dir, f"test_hf_{model_name}.py")
    if os.path.exists(target_path):
        backup_path = f"{target_path}.bak.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        shutil.copy2(target_path, backup_path)
        logger.info(f"Created backup: {backup_path}")
    
    # Prepare class name
    class_name = model_name.replace("_", " ").title().replace(" ", "")
    
    # Modify template for this model
    content = template_content
    
    # Replace model-specific values
    content = content.replace("MODEL_TYPE", model_name.upper())
    content = content.replace("model_type", model_name)
    content = content.replace("ModelClass", f"{class_name}Class")
    
    # Update model registry with this model
    registry_entry = f"""    "{model_name}": {{
        "description": "{model_name.upper()} model",
        "class": "{model_info['class']}",
        "default_model": "{model_info['model_id']}",
        "architecture": "{model_info['architecture']}"
    }},"""
    
    # Find the registry section and add this model if not already present
    registry_start = content.find("VISION_TEXT_MODELS_REGISTRY = {")
    if "SPEECH_MODELS_REGISTRY" in content:
        registry_start = content.find("SPEECH_MODELS_REGISTRY = {")
    if "ENCODER_DECODER_MODELS_REGISTRY" in content:
        registry_start = content.find("ENCODER_DECODER_MODELS_REGISTRY = {")
    
    if registry_start != -1:
        registry_end = content.find("}", registry_start)
        if registry_end != -1:
            # Check if model is already in registry
            if f'"{model_name}"' not in content[registry_start:registry_end]:
                # Add model to registry
                content = content[:registry_end] + "\n" + registry_entry + content[registry_end:]
    
    # Write the generated test file
    with open(target_path, 'w') as f:
        f.write(content)
    
    logger.info(f"Generated test file: {target_path}")
    return True

def verify_syntax(file_path):
    """Verify the syntax of a Python file."""
    try:
        result = subprocess.run(
            ["python", "-m", "py_compile", file_path],
            capture_output=True,
            text=True,
            check=False
        )
        if result.returncode == 0:
            logger.info(f"Syntax verification successful: {file_path}")
            return True
        else:
            logger.error(f"Syntax verification failed: {file_path}")
            logger.error(result.stderr)
            return False
    except Exception as e:
        logger.error(f"Error verifying syntax: {e}")
        return False

def update_architecture_types(model_name, architecture_type):
    """Update the ARCHITECTURE_TYPES dictionary in test_generator_fixed.py."""
    generator_path = os.path.join(SKILLS_DIR, "test_generator_fixed.py")
    if not os.path.exists(generator_path):
        logger.warning(f"Generator file not found: {generator_path}")
        return False
    
    try:
        with open(generator_path, 'r') as f:
            content = f.read()
        
        # Find the ARCHITECTURE_TYPES dictionary
        arch_types_start = content.find("ARCHITECTURE_TYPES = {")
        if arch_types_start == -1:
            logger.warning("ARCHITECTURE_TYPES not found in generator file")
            return False
        
        # Find the specific architecture type section
        arch_type_line = f'    "{architecture_type}": ['
        arch_type_start = content.find(arch_type_line, arch_types_start)
        if arch_type_start == -1:
            logger.warning(f"Architecture type '{architecture_type}' not found in ARCHITECTURE_TYPES")
            return False
        
        # Find the end of the architecture type list
        list_start = content.find('[', arch_type_start)
        list_end = content.find(']', list_start)
        
        # Check if model is already in the list
        arch_type_section = content[list_start:list_end]
        if f'"{model_name}"' in arch_type_section or f'"{model_name.replace("_", "-")}"' in arch_type_section:
            logger.info(f"Model '{model_name}' already exists in architecture type '{architecture_type}'")
            return True
        
        # Add the model to the list
        new_content = content[:list_end] + f', "{model_name}"' + content[list_end:]
        
        # Write the updated content
        with open(generator_path, 'w') as f:
            f.write(new_content)
        
        logger.info(f"Added '{model_name}' to architecture type '{architecture_type}'")
        return True
    except Exception as e:
        logger.error(f"Error updating ARCHITECTURE_TYPES: {e}")
        return False

def fix_manual_models(verify=True, apply=True):
    """Fix manually created model tests by regenerating them with the template system."""
    results = {
        "success": [],
        "failure": []
    }
    
    # Process each manual model
    for model_name, model_info in MANUAL_MODELS.items():
        logger.info(f"Processing model: {model_name}")
        
        # Skip if source file doesn't exist
        if not os.path.exists(model_info["source_file"]):
            logger.warning(f"Source file not found: {model_info['source_file']}")
            results["failure"].append((model_name, "Source file not found"))
            continue
        
        try:
            # Generate a new test file from the template
            success = generate_test_from_template(model_name, model_info, FIXED_TESTS_DIR)
            
            if not success:
                logger.error(f"Failed to generate test for {model_name}")
                results["failure"].append((model_name, "Template generation failed"))
                continue
            
            # Verify syntax if requested
            if verify:
                target_path = os.path.join(FIXED_TESTS_DIR, f"test_hf_{model_name}.py")
                if not verify_syntax(target_path):
                    results["failure"].append((model_name, "Syntax verification failed"))
                    continue
            
            # Update architecture types if requested
            if apply:
                update_architecture_types(model_name, model_info["architecture"])
            
            # Record success
            results["success"].append(model_name)
            logger.info(f"Successfully fixed test for {model_name}")
        
        except Exception as e:
            logger.error(f"Error fixing test for {model_name}: {e}")
            results["failure"].append((model_name, str(e)))
    
    # Print summary
    logger.info("\nFix Summary:")
    logger.info(f"- Successfully fixed: {len(results['success'])} models")
    if results["success"]:
        logger.info(f"  Models: {', '.join(results['success'])}")
    
    logger.info(f"- Failed to fix: {len(results['failure'])} models")
    if results["failure"]:
        for model, error in results["failure"]:
            logger.info(f"  - {model}: {error}")
    
    return results

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Fix manually created model tests by regenerating them with the template system"
    )
    parser.add_argument("--verify", action="store_true", 
                        help="Verify syntax after generation")
    parser.add_argument("--apply", action="store_true",
                        help="Apply changes to architecture types dictionary")
    
    args = parser.parse_args()
    
    # Fix the manual models
    results = fix_manual_models(verify=args.verify, apply=args.apply)
    
    # Success if any model was successfully fixed
    if results["success"]:
        return 0
    else:
        logger.error("Failed to fix any models")
        return 1

if __name__ == "__main__":
    sys.exit(main())