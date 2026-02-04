#!/usr/bin/env python3

"""
Expand the HuggingFace Model Registry with additional model types.

This script:
1. Adds new model types to the registry based on the ARCHITECTURE_TYPES dictionary
2. Queries the HuggingFace API for popular models of each type
3. Updates the huggingface_model_types.json file with new data
4. Verifies the integration with test_generator_fixed.py

Usage:
    python expand_model_registry.py [--all] [--dry-run] [--model-type TYPE]
"""

import os
import sys
import json
import logging
import argparse
import importlib.util
from datetime import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
CURRENT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
REGISTRY_FILE = CURRENT_DIR / "huggingface_model_types.json"

# Architecture types from find_models.py
ARCHITECTURE_TYPES = {
    "encoder-only": ["bert", "distilbert", "roberta", "electra", "camembert", "xlm-roberta", "deberta", "albert", "ernie", "rembert"],
    "decoder-only": ["gpt2", "gpt-j", "gpt-neo", "gpt-neox", "bloom", "llama", "mistral", "falcon", "phi", "mixtral", "mpt", "gemma", "opt", "codegen"],
    "encoder-decoder": ["t5", "bart", "pegasus", "mbart", "longt5", "led", "marian", "mt5", "flan", "prophetnet"],
    "vision": ["vit", "swin", "deit", "beit", "convnext", "poolformer", "dinov2", "segformer", "detr", "yolos", "mask2former"],
    "vision-text": ["vision-encoder-decoder", "vision-text-dual-encoder", "clip", "blip", "git", "flava", "paligemma"],
    "speech": ["wav2vec2", "hubert", "whisper", "bark", "speecht5", "sew", "encodec", "musicgen", "audio", "clap"],
    "multimodal": ["llava", "clip", "blip", "git", "pix2struct", "paligemma", "video-llava", "idefics", "imagebind"]
}

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

def load_registry_data():
    """Load model registry data from JSON file."""
    try:
        if os.path.exists(REGISTRY_FILE):
            with open(REGISTRY_FILE, 'r') as f:
                data = json.load(f)
                logger.info(f"Loaded model registry from {REGISTRY_FILE}")
                return data
        else:
            logger.warning(f"Registry file {REGISTRY_FILE} not found, creating new one")
            return {}
    except Exception as e:
        logger.error(f"Error loading registry file: {e}")
        return {}

def save_registry_data(data):
    """Save model registry data to JSON file."""
    try:
        with open(REGISTRY_FILE, 'w') as f:
            json.dump(data, f, indent=2)
        logger.info(f"Saved model registry to {REGISTRY_FILE}")
        return True
    except Exception as e:
        logger.error(f"Error saving registry file: {e}")
        return False

def get_all_model_types():
    """Get all model types from ARCHITECTURE_TYPES."""
    all_types = []
    for models in ARCHITECTURE_TYPES.values():
        all_types.extend(models)
    return sorted(list(set(all_types)))  # Remove duplicates

def update_model_type(model_type, dry_run=False):
    """Update a single model type in the registry."""
    try:
        # Import find_models.py
        find_models_path = CURRENT_DIR / "find_models.py"
        find_models = import_module_from_path("find_models", find_models_path)
        if not find_models:
            logger.error("Could not import find_models.py")
            return False
        
        # Load current registry data
        registry_data = load_registry_data()
        
        # Query HuggingFace API
        logger.info(f"Querying HuggingFace API for {model_type} models...")
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
            
            # Save the updated registry if not in dry run mode
            if not dry_run:
                save_registry_data(registry_data)
            else:
                logger.info(f"Dry run: Would have updated registry for {model_type}")
                
            return True
        else:
            logger.warning(f"No models found for {model_type}")
            return False
    
    except Exception as e:
        logger.error(f"Error updating model type {model_type}: {e}")
        return False

def update_all_model_types(dry_run=False):
    """Update all model types in the registry."""
    all_types = get_all_model_types()
    logger.info(f"Found {len(all_types)} model types to update")
    
    success_count = 0
    for model_type in all_types:
        if update_model_type(model_type, dry_run=dry_run):
            success_count += 1
    
    logger.info(f"Successfully updated {success_count}/{len(all_types)} model types")
    return success_count > 0

def verify_integration():
    """Verify the integration with test_generator_fixed.py."""
    try:
        # Import test_generator_fixed.py
        test_generator_path = CURRENT_DIR / "test_generator_fixed.py"
        test_generator = import_module_from_path("test_generator_fixed", test_generator_path)
        if not test_generator:
            logger.error("Could not import test_generator_fixed.py")
            return False
        
        # Check if model lookup integration is available
        has_model_lookup = hasattr(test_generator, "HAS_MODEL_LOOKUP") and getattr(test_generator, "HAS_MODEL_LOOKUP", False)
        
        # Check if get_model_from_registry function is available
        has_get_model = hasattr(test_generator, "get_model_from_registry")
        
        if has_model_lookup and has_get_model:
            logger.info("✅ Model lookup integration is properly set up in test_generator_fixed.py")
            
            # Try to get a model from the registry
            model_type = "bert"
            try:
                default_model = test_generator.get_model_from_registry(model_type)
                logger.info(f"✅ Successfully retrieved default model for {model_type}: {default_model}")
                return True
            except Exception as e:
                logger.error(f"❌ Error retrieving default model: {e}")
                return False
        else:
            logger.error("❌ Model lookup integration is not properly set up")
            logger.error(f"HAS_MODEL_LOOKUP: {has_model_lookup}, has get_model_from_registry: {has_get_model}")
            return False
    
    except Exception as e:
        logger.error(f"Error verifying integration: {e}")
        return False

def generate_summary_report():
    """Generate a summary report of the model registry."""
    try:
        # Load registry data
        registry_data = load_registry_data()
        
        if not registry_data:
            logger.warning("Registry is empty, nothing to report")
            return False
        
        # Prepare report data
        model_count = len(registry_data)
        updated_models = {model: data["updated_at"] for model, data in registry_data.items()}
        default_models = {model: data["default_model"] for model, data in registry_data.items()}
        
        # Create report
        report = "# HuggingFace Model Registry Summary\n\n"
        report += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        report += f"## Registry Statistics\n\n"
        report += f"- **Total Models**: {model_count}\n"
        
        # Count models by architecture
        arch_counts = {}
        for arch, models in ARCHITECTURE_TYPES.items():
            count = sum(1 for m in models if m in registry_data)
            arch_counts[arch] = count
            report += f"- **{arch.capitalize()}**: {count} models\n"
        
        # Default models table
        report += "\n## Default Models\n\n"
        report += "| Model Type | Default Model | Last Updated |\n"
        report += "|------------|---------------|-------------|\n"
        
        for model in sorted(registry_data.keys()):
            default = default_models.get(model, "N/A")
            updated = updated_models.get(model, "N/A")
            if isinstance(updated, str) and len(updated) > 10:
                updated = updated[:10]  # Trim ISO timestamp
            report += f"| {model} | {default} | {updated} |\n"
        
        # Write report to file
        report_path = CURRENT_DIR / "MODEL_REGISTRY_SUMMARY.md"
        with open(report_path, "w") as f:
            f.write(report)
        
        logger.info(f"Generated summary report at {report_path}")
        return True
    
    except Exception as e:
        logger.error(f"Error generating summary report: {e}")
        return False

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Expand the HuggingFace Model Registry")
    parser.add_argument("--all", action="store_true", help="Update all model types")
    parser.add_argument("--dry-run", action="store_true", help="Show changes without applying them")
    parser.add_argument("--model-type", type=str, help="Update a specific model type")
    parser.add_argument("--verify", action="store_true", help="Verify the integration with test_generator_fixed.py")
    parser.add_argument("--report", action="store_true", help="Generate a summary report")
    
    args = parser.parse_args()
    
    if args.all:
        update_all_model_types(dry_run=args.dry_run)
    elif args.model_type:
        update_model_type(args.model_type, dry_run=args.dry_run)
    
    if args.verify or args.all:
        verify_integration()
    
    if args.report or args.all:
        generate_summary_report()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())