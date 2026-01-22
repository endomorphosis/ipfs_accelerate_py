#!/usr/bin/env python3

"""
Integrate the HuggingFace model lookup system with the test generator.

This script fully integrates model lookup with test generation by:
1. Ensuring the necessary imports and functions are in test_generator_fixed.py
2. Adding model lookup support to all template files
3. Updating the defaults in MODEL_REGISTRY with values from huggingface_model_types.json
4. Running integration tests to verify the system works end-to-end

Usage:
    python integrate_test_generator.py [--all] [--verify] [--templates] [--registry]
"""

import os
import sys
import json
import logging
import argparse
import importlib.util
import subprocess
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
CURRENT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
REGISTRY_FILE = CURRENT_DIR / "huggingface_model_types.json"
TEST_GENERATOR_FILE = CURRENT_DIR / "test_generator_fixed.py"
TEMPLATES_DIR = CURRENT_DIR / "templates"

def import_module_from_path(module_name, file_path):
    """Import a module from a file path."""
    try:
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if spec is None:
            logger.warning(f"Could not find module spec for {file_path}")
            return None
            
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    except Exception as e:
        logger.warning(f"Could not import {module_name} from {file_path}: {e}")
        return None

def verify_model_lookup_integration():
    """Verify that model lookup is already integrated in test_generator_fixed.py."""
    try:
        # Import the module
        test_generator = import_module_from_path("test_generator_fixed", TEST_GENERATOR_FILE)
        if not test_generator:
            logger.error("Could not import test_generator_fixed.py")
            return False
        
        # Check for model lookup functionality
        has_model_lookup = hasattr(test_generator, "HAS_MODEL_LOOKUP")
        has_get_model = hasattr(test_generator, "get_model_from_registry")
        
        if has_model_lookup and has_get_model:
            logger.info("✅ Model lookup integration already present in test_generator_fixed.py")
            return True
        else:
            logger.warning("❌ Model lookup integration missing in test_generator_fixed.py")
            return False
    
    except Exception as e:
        logger.error(f"Error verifying model lookup integration: {e}")
        return False

def integrate_model_lookup():
    """Integrate model lookup in test_generator_fixed.py if needed."""
    if verify_model_lookup_integration():
        # Already integrated
        return True
    
    try:
        # Load the file
        with open(TEST_GENERATOR_FILE, 'r') as f:
            content = f.read()
        
        # Check for insertion point after imports
        import_section_end = content.find("# Define architecture types for model mapping")
        if import_section_end == -1:
            # Alternative: Try to find after logging setup
            import_section_end = content.find("logger = logging.getLogger(__name__)")
            if import_section_end != -1:
                # Move past the line
                import_section_end = content.find("\n", import_section_end) + 1
        
        if import_section_end == -1:
            logger.error("Could not find insertion point in test_generator_fixed.py")
            return False
        
        # Create the integration code
        lookup_integration_code = """

# Model lookup integration
try:
    from find_models import get_recommended_default_model, query_huggingface_api
    HAS_MODEL_LOOKUP = True
    logger.info("Model lookup integration available")
except ImportError:
    HAS_MODEL_LOOKUP = False
    logger.warning("Model lookup not available, using static model registry")

def get_model_from_registry(model_type):
    '''Get the best default model for a model type, using dynamic lookup if available.'''
    if HAS_MODEL_LOOKUP:
        # Try to get a recommended model from the HuggingFace API
        try:
            default_model = get_recommended_default_model(model_type)
            logger.info(f"Using recommended model for {model_type}: {default_model}")
            return default_model
        except Exception as e:
            logger.warning(f"Error getting recommended model for {model_type}: {e}")
            # Fall back to registry lookup
    
    # Use the static registry as fallback
    if model_type in MODEL_REGISTRY:
        return MODEL_REGISTRY[model_type].get("default_model")
    
    # For unknown models, use a heuristic approach
    return f"{model_type}-base" if "-base" not in model_type else model_type
"""
        
        # Insert the code
        updated_content = content[:import_section_end] + lookup_integration_code + content[import_section_end:]
        
        # Update generate_test_file to use get_model_from_registry
        old_line = "    default_model = model_config.get(\"default_model\", f\"{model_family}-base\")"
        new_line = "    default_model = get_model_from_registry(model_family)"
        
        if old_line in updated_content:
            updated_content = updated_content.replace(old_line, new_line)
            logger.info("Updated default model selection in generate_test_file function")
        else:
            # Try to find similar line
            pattern = r'(\s+)default_model\s*=\s*[^=\n]+'
            import re
            match = re.search(pattern, updated_content)
            if match:
                indent = match.group(1)
                old_line = match.group(0)
                new_line = f"{indent}default_model = get_model_from_registry(model_family)"
                updated_content = updated_content.replace(old_line, new_line)
                logger.info("Updated default model selection with pattern match")
            else:
                logger.warning("Could not find default model line to update")
        
        # Write the updated file
        with open(TEST_GENERATOR_FILE, 'w') as f:
            f.write(updated_content)
        
        logger.info(f"✅ Integrated model lookup in {TEST_GENERATOR_FILE}")
        return True
    
    except Exception as e:
        logger.error(f"Error integrating model lookup: {e}")
        return False

def update_templates():
    """Update template files to support model lookup."""
    logger.info("Updating template files to support model lookup")
    
    # Use the enhance_model_lookup.py script if available
    enhance_script = CURRENT_DIR / "enhance_model_lookup.py"
    if os.path.exists(enhance_script):
        try:
            # Run the script with --templates option
            cmd = [sys.executable, str(enhance_script), "--templates"]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info("✅ Successfully updated template files")
                return True
            else:
                logger.error(f"❌ Error updating template files: {result.stderr}")
                return False
        except Exception as e:
            logger.error(f"Error running enhance_model_lookup.py: {e}")
            return False
    else:
        logger.error(f"enhance_model_lookup.py not found at {enhance_script}")
        return False

def update_registry():
    """Update MODEL_REGISTRY in test_generator_fixed.py with values from the JSON registry."""
    logger.info("Updating MODEL_REGISTRY with values from huggingface_model_types.json")
    
    try:
        # Load the JSON registry
        if not os.path.exists(REGISTRY_FILE):
            logger.error(f"Registry file not found: {REGISTRY_FILE}")
            return False
        
        with open(REGISTRY_FILE, 'r') as f:
            registry_data = json.load(f)
        
        # Import test_generator_fixed.py
        test_generator = import_module_from_path("test_generator_fixed", TEST_GENERATOR_FILE)
        if not test_generator:
            logger.error("Could not import test_generator_fixed.py")
            return False
        
        # Get the current MODEL_REGISTRY
        current_registry = test_generator.MODEL_REGISTRY
        
        # Read the test_generator_fixed.py file
        with open(TEST_GENERATOR_FILE, 'r') as f:
            content = f.read()
        
        # Find MODEL_REGISTRY definition
        registry_start = content.find("MODEL_REGISTRY = {")
        if registry_start == -1:
            logger.error("Could not find MODEL_REGISTRY definition")
            return False
        
        # Find the end of the registry by matching braces
        registry_end = registry_start + len("MODEL_REGISTRY = {")
        brace_count = 1
        
        while brace_count > 0 and registry_end < len(content):
            if content[registry_end] == '{':
                brace_count += 1
            elif content[registry_end] == '}':
                brace_count -= 1
            registry_end += 1
        
        if brace_count != 0:
            logger.error("Could not find end of MODEL_REGISTRY definition")
            return False
        
        # Create updated registry content
        updated_registry = "MODEL_REGISTRY = {\n"
        
        # Add each model type with updated default_model if available
        for model_type, config in current_registry.items():
            updated_registry += f'    "{model_type}": {{\n'
            
            for key, value in config.items():
                # Update default_model if available in registry_data
                if key == "default_model" and model_type in registry_data:
                    new_default = registry_data[model_type].get("default_model")
                    if new_default:
                        updated_registry += f'        "{key}": "{new_default}",\n'
                        continue
                
                # Format the value based on its type
                if isinstance(value, str):
                    updated_registry += f'        "{key}": "{value}",\n'
                elif isinstance(value, list):
                    updated_registry += f'        "{key}": {value},\n'
                elif isinstance(value, dict):
                    updated_registry += f'        "{key}": {{\n'
                    for k, v in value.items():
                        if isinstance(v, dict):
                            updated_registry += f'            "{k}": {{\n'
                            for inner_k, inner_v in v.items():
                                updated_registry += f'                "{inner_k}": {inner_v},\n'
                            updated_registry += '            },\n'
                        elif isinstance(v, str):
                            updated_registry += f'            "{k}": "{v}",\n'
                        else:
                            updated_registry += f'            "{k}": {v},\n'
                    updated_registry += '        },\n'
                else:
                    updated_registry += f'        "{key}": {value},\n'
            
            updated_registry += '    },\n'
        
        updated_registry += "}"
        
        # Replace the registry in the file
        updated_content = content[:registry_start] + updated_registry + content[registry_end:]
        
        # Write the updated file
        with open(TEST_GENERATOR_FILE, 'w') as f:
            f.write(updated_content)
        
        logger.info(f"✅ Updated MODEL_REGISTRY in {TEST_GENERATOR_FILE}")
        return True
    
    except Exception as e:
        logger.error(f"Error updating registry: {e}")
        return False

def verify_integration():
    """Verify the complete integration by running tests."""
    logger.info("Verifying model lookup integration")
    
    # Use the verify_model_lookup.py script if available
    verify_script = CURRENT_DIR / "verify_model_lookup.py"
    if os.path.exists(verify_script):
        try:
            # Run the script with default options
            cmd = [sys.executable, str(verify_script), "--all"]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            # Display the output
            print(result.stdout)
            
            if result.returncode == 0:
                logger.info("✅ Integration verification successful")
                return True
            else:
                logger.error(f"❌ Integration verification failed: {result.stderr}")
                return False
        except Exception as e:
            logger.error(f"Error running verify_model_lookup.py: {e}")
            return False
    else:
        logger.error(f"verify_model_lookup.py not found at {verify_script}")
        return False

def generate_test_samples():
    """Generate test samples to verify the integration."""
    logger.info("Generating test samples")
    
    try:
        # Create output directory
        output_dir = CURRENT_DIR / "integration_samples"
        os.makedirs(output_dir, exist_ok=True)
        
        # Import test_generator_fixed
        test_generator = import_module_from_path("test_generator_fixed", TEST_GENERATOR_FILE)
        if not test_generator:
            logger.error("Could not import test_generator_fixed.py")
            return False
        
        # Generate test files for core model types
        core_types = ["bert", "gpt2", "t5", "vit"]
        success_count = 0
        
        for model_type in core_types:
            try:
                logger.info(f"Generating test for {model_type}")
                
                # Call generate_test_file
                if hasattr(test_generator, "generate_test_file"):
                    success = test_generator.generate_test_file(model_type, str(output_dir))
                    
                    if success:
                        logger.info(f"✅ Successfully generated test for {model_type}")
                        success_count += 1
                    else:
                        logger.error(f"❌ Failed to generate test for {model_type}")
                else:
                    # Direct subprocess call
                    cmd = [sys.executable, str(TEST_GENERATOR_FILE), "--generate", model_type, "--output-dir", str(output_dir)]
                    result = subprocess.run(cmd, capture_output=True, text=True)
                    
                    if result.returncode == 0:
                        logger.info(f"✅ Successfully generated test for {model_type}")
                        success_count += 1
                    else:
                        logger.error(f"❌ Failed to generate test for {model_type}: {result.stderr}")
            
            except Exception as e:
                logger.error(f"Error generating test for {model_type}: {e}")
        
        logger.info(f"Generated {success_count}/{len(core_types)} test samples")
        return success_count > 0
    
    except Exception as e:
        logger.error(f"Error generating test samples: {e}")
        return False

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Integrate model lookup with test generator")
    parser.add_argument("--all", action="store_true", help="Perform all integration steps")
    parser.add_argument("--verify", action="store_true", help="Verify integration")
    parser.add_argument("--templates", action="store_true", help="Update template files")
    parser.add_argument("--registry", action="store_true", help="Update MODEL_REGISTRY with registry values")
    parser.add_argument("--samples", action="store_true", help="Generate test samples")
    
    args = parser.parse_args()
    
    # Check if model lookup is already integrated
    is_integrated = verify_model_lookup_integration()
    
    # Perform requested actions
    if not is_integrated or args.all:
        integrate_model_lookup()
    
    if args.all or args.templates:
        update_templates()
    
    if args.all or args.registry:
        update_registry()
    
    if args.all or args.samples:
        generate_test_samples()
    
    if args.all or args.verify:
        verify_integration()
    
    logger.info("Model lookup integration complete")
    return 0

if __name__ == "__main__":
    sys.exit(main())