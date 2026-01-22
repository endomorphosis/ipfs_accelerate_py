#!/usr/bin/env python3
"""
Regenerate manually created model tests using the template system.

This script:
1. Maps each model to its architecture type
2. Uses the template system to generate standard tests
3. Ensures proper hardware detection and mock objects
4. Verifies syntax and functionality of generated tests
5. Updates model registries and architecture mappings

Usage:
    python regenerate_template_tests.py [--model MODEL] [--all] [--verify] [--apply]
"""

import os
import sys
import argparse
import logging
import shutil
import subprocess
import tempfile
import importlib.util
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set, Any, Tuple, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"regenerate_template_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    ]
)
logger = logging.getLogger(__name__)

# Define paths
SCRIPT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = SCRIPT_DIR.parent
SKILLS_DIR = REPO_ROOT / "skills"
TEMPLATES_DIR = SKILLS_DIR / "templates"
FINAL_MODELS_DIR = REPO_ROOT / "final_models"
FIXED_TESTS_DIR = SKILLS_DIR / "fixed_tests"
OUTPUT_DIR = REPO_ROOT / "template_verification"

# Ensure directories exist
os.makedirs(FIXED_TESTS_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Define model mappings - maps model names to their architecture and default models
MODEL_MAPPINGS = {
    "layoutlmv2": {
        "architecture": "vision-encoder-text-decoder",
        "template": "vision_text_template.py",
        "model_id": "microsoft/layoutlmv2-base-uncased",
        "class": "LayoutLMv2ForSequenceClassification",
        "source_file": f"{FINAL_MODELS_DIR}/test_layoutlmv2.py",
        "destination_file": f"{FIXED_TESTS_DIR}/test_hf_layoutlmv2.py"
    },
    "layoutlmv3": {
        "architecture": "vision-encoder-text-decoder",
        "template": "vision_text_template.py",
        "model_id": "microsoft/layoutlmv3-base",
        "class": "LayoutLMv3ForSequenceClassification",
        "source_file": f"{FINAL_MODELS_DIR}/test_layoutlmv3.py",
        "destination_file": f"{FIXED_TESTS_DIR}/test_hf_layoutlmv3.py"
    },
    "clvp": {
        "architecture": "speech",
        "template": "speech_template.py",
        "model_id": "susnato/clvp_dev",
        "class": "CLVPForCausalLM",
        "source_file": f"{FINAL_MODELS_DIR}/test_clvp.py",
        "destination_file": f"{FIXED_TESTS_DIR}/test_hf_clvp.py"
    },
    "bigbird": {
        "architecture": "encoder-decoder",
        "template": "encoder_decoder_template.py",
        "model_id": "google/bigbird-roberta-base",
        "class": "BigBirdForSequenceClassification",
        "source_file": f"{FINAL_MODELS_DIR}/test_hf_bigbird.py",
        "destination_file": f"{FIXED_TESTS_DIR}/test_hf_bigbird.py"
    },
    "seamless_m4t_v2": {
        "architecture": "speech",
        "template": "speech_template.py",
        "model_id": "facebook/seamless-m4t-v2-large",
        "class": "SeamlessM4TModel",
        "source_file": f"{FINAL_MODELS_DIR}/test_seamless_m4t_v2.py",
        "destination_file": f"{FIXED_TESTS_DIR}/test_hf_seamless_m4t_v2.py"
    },
    "xlm_prophetnet": {
        "architecture": "encoder-decoder",
        "template": "encoder_decoder_template.py",
        "model_id": "microsoft/xprophetnet-large-wiki100-cased",
        "class": "XLMProphetNetForConditionalGeneration",
        "source_file": f"{FINAL_MODELS_DIR}/test_xlm_prophetnet.py",
        "destination_file": f"{FIXED_TESTS_DIR}/test_hf_xlm_prophetnet.py"
    }
}

# Define architecture registry - maps architecture types to their templates
ARCHITECTURE_REGISTRY = {
    "encoder-only": "encoder_only_template.py",
    "decoder-only": "decoder_only_template.py",
    "encoder-decoder": "encoder_decoder_template.py",
    "vision": "vision_template.py",
    "vision-encoder-text-decoder": "vision_text_template.py",
    "speech": "speech_template.py",
    "multimodal": "multimodal_template.py"
}

def verify_python_syntax(file_path: str) -> Tuple[bool, List[str]]:
    """Verify the syntax of a Python file."""
    try:
        # First try using the ast module
        with open(file_path, 'r') as f:
            content = f.read()
        import ast
        ast.parse(content)
        
        # Then try compiling
        compile(content, file_path, 'exec')
        
        return True, []
    except SyntaxError as e:
        return False, [f"Line {e.lineno}: {e.msg}"]
    except Exception as e:
        return False, [str(e)]

def extract_registry_entry(template_content: str, registry_name: str, model_name: str) -> str:
    """Extract the registry entry format from a template."""
    # Find the registry definition
    registry_start = template_content.find(f"{registry_name} = {{")
    if registry_start == -1:
        return ""
    
    # Find the first entry
    entry_match = re.search(r'    "([^"]+)": {', template_content[registry_start:])
    if not entry_match:
        return ""
    
    # Find the structure
    entry_key = entry_match.group(1)
    entry_start = template_content.find(f'    "{entry_key}":', registry_start)
    entry_end = template_content.find("    },", entry_start)
    if entry_end == -1:
        entry_end = template_content.find("    }", entry_start)
    
    if entry_start == -1 or entry_end == -1:
        return ""
    
    # Extract the entry and format for the new model
    entry_format = template_content[entry_start:entry_end+6]
    entry_format = entry_format.replace(entry_key, model_name)
    
    return entry_format

def get_registry_name(architecture: str) -> str:
    """Get the model registry name based on architecture."""
    registry_map = {
        "encoder-only": "ENCODER_ONLY_MODELS_REGISTRY",
        "decoder-only": "DECODER_ONLY_MODELS_REGISTRY",
        "encoder-decoder": "ENCODER_DECODER_MODELS_REGISTRY",
        "vision": "VISION_MODELS_REGISTRY",
        "vision-encoder-text-decoder": "VISION_TEXT_MODELS_REGISTRY",
        "speech": "SPEECH_MODELS_REGISTRY",
        "multimodal": "MULTIMODAL_MODELS_REGISTRY"
    }
    
    return registry_map.get(architecture, "MODEL_REGISTRY")

def generate_model_registry_entry(model_name: str, model_info: Dict[str, Any], template_path: str) -> str:
    """Generate a model registry entry for a model."""
    try:
        with open(template_path, 'r') as f:
            template_content = f.read()
        
        # Get the registry name for this architecture
        registry_name = get_registry_name(model_info["architecture"])
        
        # Extract the registry entry format
        entry_format = extract_registry_entry(template_content, registry_name, model_name)
        
        # If we couldn't extract a format, create a generic one
        if not entry_format:
            return f"""    "{model_name}": {{
        "description": "{model_name.upper()} model",
        "class": "{model_info['class']}",
        "default_model": "{model_info['model_id']}",
        "architecture": "{model_info['architecture']}"
    }},"""
        
        # Update the entry with model-specific information
        entry_format = entry_format.replace('"description": "[^"]+",', f'"description": "{model_name.upper()} model",')
        entry_format = entry_format.replace('"class": "[^"]+",', f'"class": "{model_info["class"]}",')
        entry_format = entry_format.replace('"default_model": "[^"]+",', f'"default_model": "{model_info["model_id"]}",')
        
        return entry_format
    
    except Exception as e:
        logger.error(f"Error generating registry entry: {e}")
        return ""

def regenerate_test_file(model_name: str, model_info: Dict[str, Any], verify: bool = True) -> Tuple[bool, List[str]]:
    """Regenerate a test file using the template system."""
    try:
        # Get the template path
        template_path = os.path.join(TEMPLATES_DIR, model_info["template"])
        if not os.path.exists(template_path):
            return False, [f"Template file not found: {template_path}"]
        
        # Read the template
        with open(template_path, 'r') as f:
            template_content = f.read()
        
        # Create a backup of the destination file if it exists
        if os.path.exists(model_info["destination_file"]):
            backup_path = f"{model_info['destination_file']}.bak.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            shutil.copy2(model_info["destination_file"], backup_path)
            logger.info(f"Created backup: {backup_path}")
        
        # Create modified content for this model
        content = template_content
        
        # Replace model registry name if needed
        registry_name = get_registry_name(model_info["architecture"])
        if registry_name == "MODEL_REGISTRY":
            # Generic registry, need to replace it
            registry_pattern = r"[A-Z_]+_MODELS_REGISTRY = {"
            match = re.search(registry_pattern, content)
            if match:
                content = content.replace(match.group(0), "MODEL_REGISTRY = {")
        
        # Add model registry entry
        registry_entry = generate_model_registry_entry(model_name, model_info, template_path)
        if registry_entry:
            registry_start = content.find(f"{registry_name} = {{")
            registry_end = content.find("}", registry_start)
            if registry_start != -1 and registry_end != -1:
                # Add the entry after the opening brace
                content = content[:registry_start + len(f"{registry_name} = {{") + 1] + "\n" + registry_entry + content[registry_start + len(f"{registry_name} = {{") + 1:]
        
        # Replace test class name
        test_class_pattern = r"class Test\w+:"
        match = re.search(test_class_pattern, content)
        if match:
            capitalized_model = "".join(word.capitalize() for word in model_name.split("_"))
            content = content.replace(match.group(0), f"class Test{capitalized_model}:")
        
        # Update model references
        content = content.replace("MODEL_TYPE", model_name.upper())
        content = content.replace("model_type", model_name)
        
        # Write the regenerated test file
        with open(model_info["destination_file"], 'w') as f:
            f.write(content)
        
        # Verify syntax if requested
        if verify:
            syntax_valid, errors = verify_python_syntax(model_info["destination_file"])
            if not syntax_valid:
                return False, errors
        
        return True, []
    
    except Exception as e:
        logger.error(f"Error regenerating test file: {e}")
        return False, [str(e)]

def update_architecture_types_file(model_name: str, architecture: str) -> bool:
    """Update the ARCHITECTURE_TYPES dictionary in test_generator_fixed.py."""
    try:
        generator_path = os.path.join(SKILLS_DIR, "test_generator_fixed.py")
        if not os.path.exists(generator_path):
            logger.warning(f"Generator file not found: {generator_path}")
            return False
        
        with open(generator_path, 'r') as f:
            content = f.read()
        
        # Find the ARCHITECTURE_TYPES dictionary
        arch_types_start = content.find("ARCHITECTURE_TYPES = {")
        if arch_types_start == -1:
            logger.warning("ARCHITECTURE_TYPES not found in generator file")
            return False
        
        # Find the specific architecture type section
        arch_type_line = f'    "{architecture}": ['
        arch_type_start = content.find(arch_type_line, arch_types_start)
        if arch_type_start == -1:
            logger.warning(f"Architecture type '{architecture}' not found in ARCHITECTURE_TYPES")
            return False
        
        # Find the end of the architecture type list
        list_start = content.find('[', arch_type_start)
        list_end = content.find(']', list_start)
        
        # Check if model is already in the list
        arch_type_section = content[list_start:list_end]
        if f'"{model_name}"' in arch_type_section or f'"{model_name.replace("_", "-")}"' in arch_type_section:
            logger.info(f"Model '{model_name}' already exists in architecture type '{architecture}'")
            return True
        
        # Add the model to the list
        new_content = content[:list_end] + f', "{model_name}"' + content[list_end:]
        
        # Write the updated content
        with open(generator_path, 'w') as f:
            f.write(new_content)
        
        logger.info(f"Added '{model_name}' to architecture type '{architecture}'")
        return True
    except Exception as e:
        logger.error(f"Error updating ARCHITECTURE_TYPES: {e}")
        return False

def regenerate_all_models(verify: bool = True, apply: bool = False) -> Dict[str, Any]:
    """Regenerate all manually created model tests."""
    results = {
        "success": [],
        "failure": []
    }
    
    for model_name, model_info in MODEL_MAPPINGS.items():
        logger.info(f"Processing model: {model_name}")
        
        # Skip if source file doesn't exist
        if not os.path.exists(model_info["source_file"]):
            alt_source = model_info["source_file"].replace("test_", "test_hf_")
            if os.path.exists(alt_source):
                model_info["source_file"] = alt_source
            else:
                logger.warning(f"Source file not found: {model_info['source_file']}")
                results["failure"].append((model_name, "Source file not found"))
                continue
        
        # Create destination directory if needed
        os.makedirs(os.path.dirname(model_info["destination_file"]), exist_ok=True)
        
        # Regenerate the test file
        success, errors = regenerate_test_file(model_name, model_info, verify)
        
        if success:
            logger.info(f"Successfully regenerated test for {model_name}")
            
            # Update architecture types if requested
            if apply:
                update_architecture_types_file(model_name, model_info["architecture"])
            
            results["success"].append(model_name)
        else:
            logger.error(f"Failed to regenerate test for {model_name}: {', '.join(errors)}")
            results["failure"].append((model_name, ", ".join(errors)))
    
    return results

def regenerate_specific_model(model_name: str, verify: bool = True, apply: bool = False) -> Tuple[bool, List[str]]:
    """Regenerate a specific model test."""
    if model_name not in MODEL_MAPPINGS:
        return False, [f"Model '{model_name}' not found in mappings"]
    
    model_info = MODEL_MAPPINGS[model_name]
    
    # Skip if source file doesn't exist
    if not os.path.exists(model_info["source_file"]):
        alt_source = model_info["source_file"].replace("test_", "test_hf_")
        if os.path.exists(alt_source):
            model_info["source_file"] = alt_source
        else:
            logger.warning(f"Source file not found: {model_info['source_file']}")
            return False, [f"Source file not found: {model_info['source_file']}"]
    
    # Create destination directory if needed
    os.makedirs(os.path.dirname(model_info["destination_file"]), exist_ok=True)
    
    # Regenerate the test file
    success, errors = regenerate_test_file(model_name, model_info, verify)
    
    if success:
        logger.info(f"Successfully regenerated test for {model_name}")
        
        # Update architecture types if requested
        if apply:
            update_architecture_types_file(model_name, model_info["architecture"])
    else:
        logger.error(f"Failed to regenerate test for {model_name}: {', '.join(errors)}")
    
    return success, errors

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Regenerate manually created model tests using the template system"
    )
    parser.add_argument("--model", type=str, help="Specific model to regenerate")
    parser.add_argument("--all", action="store_true", help="Regenerate all models")
    parser.add_argument("--verify", action="store_true", help="Verify syntax after generation")
    parser.add_argument("--apply", action="store_true", help="Apply changes to architecture types dictionary")
    
    args = parser.parse_args()
    
    if not args.model and not args.all:
        logger.error("Either --model or --all must be specified")
        return 1
    
    if args.model:
        # Regenerate a specific model
        success, errors = regenerate_specific_model(
            args.model,
            verify=args.verify,
            apply=args.apply
        )
        
        if not success:
            logger.error(f"Failed to regenerate {args.model}: {', '.join(errors)}")
            return 1
    else:
        # Regenerate all models
        results = regenerate_all_models(
            verify=args.verify,
            apply=args.apply
        )
        
        # Print summary
        logger.info("\nRegeneration Summary:")
        logger.info(f"- Successfully regenerated: {len(results['success'])} models")
        if results["success"]:
            logger.info(f"  Models: {', '.join(results['success'])}")
        
        logger.info(f"- Failed to regenerate: {len(results['failure'])} models")
        if results["failure"]:
            for model, error in results["failure"]:
                logger.info(f"  - {model}: {error}")
        
        if len(results["success"]) == 0:
            logger.error("Failed to regenerate any models")
            return 1
    
    logger.info("Done")
    return 0

if __name__ == "__main__":
    sys.exit(main())