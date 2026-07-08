#!/usr/bin/env python3

"""
Integrates the HuggingFace model lookup system with the test generator.

This script:
1. Updates the test_generator_fixed.py to use find_models.py for model lookup
2. Enhances the MODEL_REGISTRY with entries from the huggingface_model_types.json
3. Adds dynamic model lookup capabilities to test generation

Usage:
    python integrate_model_lookup.py [--update-all] [--dry-run]
"""

import os
import sys
import json
import logging
import argparse
import importlib.util
from pathlib import Path
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
CURRENT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))

def load_module_from_path(module_name, file_path):
    """Dynamically load a module from a file path."""
    try:
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    except Exception as e:
        logger.error(f"Error loading module {module_name} from {file_path}: {e}")
        return None

def load_registry_data(registry_file=None):
    """Load model registry data from JSON file."""
    if registry_file is None:
        registry_file = CURRENT_DIR / "huggingface_model_types.json"
    
    try:
        if os.path.exists(registry_file):
            with open(registry_file, 'r') as f:
                data = json.load(f)
                logger.info(f"Loaded model registry from {registry_file}")
                return data
        else:
            logger.warning(f"Registry file {registry_file} not found")
            return {}
    except Exception as e:
        logger.error(f"Error loading registry file: {e}")
        return {}

def update_generator_with_model_lookup():
    """Update the test generator to use find_models.py for model lookup."""
    generator_file = CURRENT_DIR / "test_generator_fixed.py"
    
    try:
        with open(generator_file, 'r') as f:
            generator_code = f.read()
        
        # Check if integration already exists
        if "from find_models import get_recommended_default_model" in generator_code:
            logger.info("Model lookup integration already exists in test generator")
            return True
        
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
        
        # Insert code after the imports
        import_section_end = generator_code.find("# Forward declarations for indentation fixing functions")
        if import_section_end == -1:
            # Fallback: find after the ARCHITECTURE_TYPES dict
            import_section_end = generator_code.find("# Configure logging")
        
        if import_section_end != -1:
            # Insert the integration code
            updated_code = (generator_code[:import_section_end] + 
                           lookup_integration_code + 
                           generator_code[import_section_end:])
            
            # Update generate_test_file to use the new function
            original_model_line = "    default_model = model_config.get(\"default_model\", f\"{model_family}-base\")"
            updated_model_line = "    default_model = get_model_from_registry(model_family)"
            
            if original_model_line in updated_code:
                updated_code = updated_code.replace(original_model_line, updated_model_line)
                logger.info("Updated default model selection in generate_test_file function")
            else:
                logger.warning("Could not find default model line in generator code")
            
            # Write the updated code back
            with open(generator_file, 'w') as f:
                f.write(updated_code)
            
            logger.info(f"Updated {generator_file} with model lookup integration")
            return True
        else:
            logger.error("Could not find appropriate insertion point in generator code")
            return False
            
    except Exception as e:
        logger.error(f"Error updating generator with model lookup: {e}")
        return False

def update_model_registry_from_json(registry_file=None, generator_file=None, dry_run=False):
    """Update MODEL_REGISTRY in test_generator_fixed.py with data from JSON registry."""
    if registry_file is None:
        registry_file = CURRENT_DIR / "huggingface_model_types.json"
    
    if generator_file is None:
        generator_file = CURRENT_DIR / "test_generator_fixed.py"
    
    try:
        # Load registry data
        registry_data = load_registry_data(registry_file)
        if not registry_data:
            logger.error("No registry data to update with")
            return False
        
        # Load current generator code
        with open(generator_file, 'r') as f:
            generator_code = f.read()
        
        # Find MODEL_REGISTRY declaration
        registry_start = generator_code.find("MODEL_REGISTRY = {")
        if registry_start == -1:
            logger.error("Could not find MODEL_REGISTRY in generator code")
            return False
        
        # Find the end of the registry
        registry_end = generator_code.find("}", registry_start)
        registry_depth = 1
        while registry_depth > 0 and registry_end < len(generator_code) - 1:
            registry_end += 1
            if generator_code[registry_end] == '{':
                registry_depth += 1
            elif generator_code[registry_end] == '}':
                registry_depth -= 1
        
        # Extract current registry
        registry_str = generator_code[registry_start:registry_end+1]
        
        # Load test_generator_fixed.py as a module to get the current MODEL_REGISTRY
        test_generator = load_module_from_path("test_generator_fixed", generator_file)
        if not test_generator:
            logger.error("Could not load test_generator_fixed.py as a module")
            return False
        
        current_registry = test_generator.MODEL_REGISTRY
        
        # Update the registry with new default models
        updated_registry = current_registry.copy()
        changes_made = False
        
        for model_type, data in registry_data.items():
            if model_type in updated_registry:
                # Only update the default_model, preserve other configurations
                old_default = updated_registry[model_type].get("default_model")
                new_default = data.get("default_model")
                
                if old_default != new_default and new_default:
                    logger.info(f"Updating {model_type} default model: {old_default} -> {new_default}")
                    updated_registry[model_type]["default_model"] = new_default
                    changes_made = True
        
        # If no changes, exit early
        if not changes_made:
            logger.info("No changes needed to MODEL_REGISTRY")
            return True
        
        # Create the updated registry string
        updated_registry_str = "MODEL_REGISTRY = {\n"
        for model_type, config in updated_registry.items():
            updated_registry_str += f'    "{model_type}": {{\n'
            for key, value in config.items():
                if isinstance(value, str):
                    updated_registry_str += f'        "{key}": "{value}",\n'
                elif isinstance(value, dict):
                    updated_registry_str += f'        "{key}": {{\n'
                    for k, v in value.items():
                        if isinstance(v, dict):
                            updated_registry_str += f'            "{k}": {{\n'
                            for inner_k, inner_v in v.items():
                                updated_registry_str += f'                "{inner_k}": {inner_v},\n'
                            updated_registry_str += '            },\n'
                        elif isinstance(v, str):
                            updated_registry_str += f'            "{k}": "{v}",\n'
                        else:
                            updated_registry_str += f'            "{k}": {v},\n'
                    updated_registry_str += '        },\n'
                else:
                    updated_registry_str += f'        "{key}": {value},\n'
            updated_registry_str += '    },\n'
        updated_registry_str += "}"
        
        # Only update if not in dry-run mode
        if not dry_run:
            # Replace the registry in the file
            updated_code = generator_code[:registry_start] + updated_registry_str + generator_code[registry_end+1:]
            
            with open(generator_file, 'w') as f:
                f.write(updated_code)
            
            logger.info(f"Updated MODEL_REGISTRY in {generator_file}")
        else:
            logger.info("Dry run: Would have updated MODEL_REGISTRY")
        
        return True
    
    except Exception as e:
        logger.error(f"Error updating model registry: {e}")
        return False

def update_all_models_in_registry(dry_run=False):
    """Update all model types in the registry with data from the HuggingFace API."""
    try:
        # Import find_models.py
        find_models = load_module_from_path("find_models", CURRENT_DIR / "find_models.py")
        if not find_models:
            logger.error("Could not load find_models.py")
            return False
        
        # Load test_generator_fixed.py to get model list
        test_generator = load_module_from_path("test_generator_fixed", CURRENT_DIR / "test_generator_fixed.py")
        if not test_generator:
            logger.error("Could not load test_generator_fixed.py")
            return False
        
        model_types = list(test_generator.MODEL_REGISTRY.keys())
        logger.info(f"Found {len(model_types)} model types in MODEL_REGISTRY")
        
        # Load current registry data
        registry_data = load_registry_data()
        
        # Query each model type
        success_count = 0
        for model_type in model_types:
            logger.info(f"Querying HuggingFace API for {model_type} models...")
            
            try:
                # Query HuggingFace API
                popular_models = find_models.query_huggingface_api(model_type, limit=10)
                
                if popular_models:
                    # Get recommended default model
                    default_model = find_models.get_recommended_default_model(model_type)
                    
                    # Update registry
                    registry_data[model_type] = {
                        "default_model": default_model,
                        "models": [m["id"] for m in popular_models],
                        "downloads": {m["id"]: m.get("downloads", 0) for m in popular_models},
                        "updated_at": datetime.now().isoformat()
                    }
                    
                    logger.info(f"Updated registry for {model_type} with default model: {default_model}")
                    success_count += 1
            except Exception as e:
                logger.error(f"Error processing {model_type}: {e}")
        
        # Save the updated registry if not in dry run mode
        if not dry_run:
            registry_file = CURRENT_DIR / "huggingface_model_types.json"
            with open(registry_file, 'w') as f:
                json.dump(registry_data, f, indent=2)
            logger.info(f"Saved updated registry to {registry_file}")
        else:
            logger.info(f"Dry run: Would have updated registry for {success_count} models")
        
        # Now update the MODEL_REGISTRY in test_generator_fixed.py
        if success_count > 0:
            update_model_registry_from_json(dry_run=dry_run)
        
        logger.info(f"Updated {success_count}/{len(model_types)} model types in registry")
        return success_count > 0
    
    except Exception as e:
        logger.error(f"Error updating all models: {e}")
        return False

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Integrate HuggingFace model lookup with test generator")
    parser.add_argument("--update-all", action="store_true", help="Update all model types in the registry")
    parser.add_argument("--dry-run", action="store_true", help="Show changes without applying them")
    parser.add_argument("--model", type=str, help="Update a specific model type in the registry")
    
    args = parser.parse_args()
    
    # Update test generator with model lookup integration
    if not args.dry_run:
        success = update_generator_with_model_lookup()
        if not success:
            logger.error("Failed to update test generator with model lookup integration")
            return 1
    
    # Update all models if requested
    if args.update_all:
        success = update_all_models_in_registry(dry_run=args.dry_run)
        if not success:
            logger.error("Failed to update all models in registry")
            return 1
    elif args.model:
        # Import find_models.py
        find_models = load_module_from_path("find_models", CURRENT_DIR / "find_models.py")
        if not find_models:
            logger.error("Could not load find_models.py")
            return 1
        
        # Load current registry data
        registry_data = load_registry_data()
        
        # Query specific model type
        logger.info(f"Querying HuggingFace API for {args.model} models...")
        try:
            # Query HuggingFace API
            popular_models = find_models.query_huggingface_api(args.model, limit=10)
            
            if popular_models:
                # Get recommended default model
                default_model = find_models.get_recommended_default_model(args.model)
                
                # Update registry
                registry_data[args.model] = {
                    "default_model": default_model,
                    "models": [m["id"] for m in popular_models],
                    "downloads": {m["id"]: m.get("downloads", 0) for m in popular_models},
                    "updated_at": datetime.now().isoformat()
                }
                
                logger.info(f"Updated registry for {args.model} with default model: {default_model}")
                
                # Save the updated registry if not in dry run mode
                if not args.dry_run:
                    registry_file = CURRENT_DIR / "huggingface_model_types.json"
                    with open(registry_file, 'w') as f:
                        json.dump(registry_data, f, indent=2)
                    logger.info(f"Saved updated registry to {registry_file}")
                else:
                    logger.info(f"Dry run: Would have updated registry for {args.model}")
                
                # Now update the MODEL_REGISTRY in test_generator_fixed.py
                update_model_registry_from_json(dry_run=args.dry_run)
            else:
                logger.error(f"No models found for {args.model}")
                return 1
        except Exception as e:
            logger.error(f"Error processing {args.model}: {e}")
            return 1
    else:
        # Just update the registry with existing data
        success = update_model_registry_from_json(dry_run=args.dry_run)
        if not success:
            logger.error("Failed to update model registry")
            return 1
    
    logger.info("Integration completed successfully")
    return 0

if __name__ == "__main__":
    sys.exit(main())