#!/usr/bin/env python3

"""
Example script that demonstrates how to use the model registry data directly.

This script:
1. Loads the HuggingFace model registry data
2. Gets default models for testing
3. Shows how to use the data for various model types

Usage:
    python use_model_registry.py [--model MODEL_TYPE]
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
CURRENT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
REGISTRY_FILE = CURRENT_DIR / "huggingface_model_types.json"

def load_registry():
    """Load the model registry data from JSON file."""
    try:
        if os.path.exists(REGISTRY_FILE):
            with open(REGISTRY_FILE, 'r') as f:
                data = json.load(f)
                logger.info(f"Loaded model registry from {REGISTRY_FILE}")
                return data
        else:
            logger.warning(f"Registry file {REGISTRY_FILE} not found")
            return {}
    except Exception as e:
        logger.error(f"Error loading registry file: {e}")
        return {}

def get_default_model(model_type):
    """Get the recommended default model for a model type."""
    registry = load_registry()
    
    if model_type in registry:
        default_model = registry[model_type].get("default_model")
        logger.info(f"Found recommended model for {model_type} in registry: {default_model}")
        return default_model
    
    # Fallback to hardcoded defaults if not in registry
    fallback_defaults = {
        "bert": "bert-base-uncased",
        "gpt2": "gpt2",
        "t5": "t5-small",
        "vit": "google/vit-base-patch16-224",
        "gpt-j": "EleutherAI/gpt-j-6b",
        "xlm-roberta": "xlm-roberta-base",
        "whisper": "openai/whisper-tiny"
    }
    
    default_model = fallback_defaults.get(model_type, f"{model_type}-base")
    logger.info(f"Using fallback default model for {model_type}: {default_model}")
    return default_model

def show_registry_info():
    """Show information about the registry."""
    registry = load_registry()
    
    if not registry:
        print("No registry data available.")
        return
    
    print(f"\nRegistry contains {len(registry)} model types:")
    
    for model_type, data in sorted(registry.items()):
        default_model = data.get("default_model", "Unknown")
        model_count = len(data.get("models", []))
        updated_at = data.get("updated_at", "Unknown")
        
        print(f"  - {model_type}: Default model = {default_model}")
        print(f"    Models: {model_count} | Updated: {updated_at}")
        
        # Show top 3 models by downloads if available
        if "downloads" in data and data["downloads"]:
            downloads = data["downloads"]
            top_models = sorted(downloads.items(), key=lambda x: x[1], reverse=True)[:3]
            
            print("    Top models by downloads:")
            for model, count in top_models:
                print(f"      - {model}: {count:,} downloads")
        
        print()

def get_model_info(model_type):
    """Get detailed information about a specific model type."""
    registry = load_registry()
    
    if model_type not in registry:
        print(f"Model type '{model_type}' not found in registry.")
        return
    
    data = registry[model_type]
    default_model = data.get("default_model", "Unknown")
    models = data.get("models", [])
    downloads = data.get("downloads", {})
    updated_at = data.get("updated_at", "Unknown")
    
    print(f"\nInformation for model type: {model_type}")
    print(f"Default model: {default_model}")
    print(f"Updated at: {updated_at}")
    print(f"Available models: {len(models)}")
    
    if downloads:
        print("\nModels by download count:")
        for model, count in sorted(downloads.items(), key=lambda x: x[1], reverse=True):
            print(f"  - {model}: {count:,} downloads")
    
    # Recommended default model
    print(f"\nRecommended default model for testing: {get_default_model(model_type)}")

def main():
    parser = argparse.ArgumentParser(description="Use the HuggingFace model registry")
    parser.add_argument("--model", type=str, help="Show information for a specific model type")
    parser.add_argument("--list", action="store_true", help="List all model types in the registry")
    
    args = parser.parse_args()
    
    if args.model:
        get_model_info(args.model)
    elif args.list or len(sys.argv) == 1:
        show_registry_info()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())