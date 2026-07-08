#!/usr/bin/env python3

"""
Find and categorize models from the HuggingFace transformers library.

This script:
1. Extracts all model classes from the transformers library using introspection
2. Categorizes models by architecture type (encoder-only, decoder-only, etc.)
3. Identifies hyphenated model names that require special handling
4. Generates configuration entries that can be added to the MODEL_REGISTRY
5. Prioritizes models based on usage and importance (Tier 1, 2, and 3)
6. Outputs a report of model coverage and gaps

Usage:
    python find_models.py [--hyphenated-only] [--max-models N] [--output OUTPUT_FILE]
"""

import os
import sys
import inspect
import json
import logging
import argparse
import re
import subprocess
from importlib import import_module
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Set

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Model architecture categories
ARCHITECTURE_TYPES = {
    "encoder-only": ["bert", "distilbert", "roberta", "electra", "camembert", "xlm-roberta", "deberta", "albert", "ernie", "rembert"],
    "decoder-only": ["gpt2", "gpt-j", "gpt-neo", "gpt-neox", "bloom", "llama", "mistral", "falcon", "phi", "mixtral", "mpt", "gemma", "opt", "codegen"],
    "encoder-decoder": ["t5", "bart", "pegasus", "mbart", "longt5", "led", "marian", "mt5", "flan", "prophetnet"],
    "vision": ["vit", "swin", "deit", "beit", "convnext", "poolformer", "dinov2", "segformer", "detr", "yolos", "mask2former"],
    "vision-text": ["vision-encoder-decoder", "vision-text-dual-encoder", "clip", "blip", "git", "flava", "paligemma"],
    "speech": ["wav2vec2", "hubert", "whisper", "bark", "speecht5", "sew", "encodec", "musicgen", "audio", "clap"],
    "multimodal": ["llava", "clip", "blip", "git", "pix2struct", "paligemma", "video-llava", "idefics", "imagebind"]
}

# Class name capitalization patterns, example: GPT2, ViT, CLIP, BEiT, etc.
CLASS_NAME_CAPITALIZATION = {
    "gpt2": "GPT2",
    "gpt-j": "GPTJ",
    "gpt-neo": "GPTNeo",
    "gpt-neox": "GPTNeoX",
    "t5": "T5",
    "mt5": "MT5",
    "vit": "ViT",
    "deit": "DeiT",
    "beit": "BEiT",
    "clip": "CLIP",
    "blip": "BLIP",
    "bert": "BERT",
    "roberta": "RoBERTa",
    "distilbert": "DistilBERT",
    "bart": "BART",
    "mbart": "MBART",
    "xlm-roberta": "XLMRoBERTa",
    "opt": "OPT",
    "llama": "Llama",
    "led": "LED",
    "flava": "FLAVA",
    "git": "GIT",
    "swin": "Swin",
    "sam": "SAM",
    "hubert": "HuBERT",
    "wav2vec2": "Wav2Vec2",
    "clap": "CLAP",
    "dinov2": "DINOv2",
    "encodec": "EnCodec",
    "musicgen": "MusicGen",
    "whisper": "Whisper",
    "llava": "LLaVA",
    "idefics": "IDEFICS",
    "paligemma": "PaliGemma",
    "imagebind": "ImageBind",
    "convnext": "ConvNeXT",
    "segformer": "SegFormer",
    "mask2former": "Mask2Former"
}

def try_model_access(model_name):
    """Try to access a model's info using Hugging Face Hub."""
    try:
        # Use python subprocess to prevent crashing if huggingface_hub is not available
        output = subprocess.check_output([
            sys.executable,
            "-c",
            f"from huggingface_hub import model_info; "
            f"info = model_info('{model_name}', token=None, timeout=5); "
            f"print(json.dumps({{'id': info.id, 'private': info.private, 'sha': info.sha[:8] if info.sha else None, 'tags': list(info.tags)[:3] if info.tags else []}}))"
        ], stderr=subprocess.PIPE, text=True, timeout=10)
        return json.loads(output.strip())
    except subprocess.CalledProcessError as e:
        return {"error": e.stderr.strip(), "id": model_name, "private": True}
    except Exception as e:
        return {"error": str(e), "id": model_name, "private": True}

def query_huggingface_api(model_type, limit=5, min_downloads=1000):
    """Query the HuggingFace API for popular models of a specific type."""
    try:
        import requests
        url = f"https://huggingface.co/api/models?filter={model_type}&sort=downloads&direction=-1&limit={limit}"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        models = response.json()
        
        # Filter models by download count if specified
        if min_downloads > 0:
            models = [m for m in models if m.get("downloads", 0) >= min_downloads]
        
        # Extract relevant information
        model_info = []
        for model in models:
            model_id = model.get("id")
            downloads = model.get("downloads", 0)
            likes = model.get("likes", 0)
            tags = model.get("tags", [])
            model_info.append({
                "id": model_id,
                "downloads": downloads,
                "likes": likes,
                "tags": tags,
                "model_type": model_type
            })
        
        logger.info(f"Found {len(model_info)} popular models for {model_type}")
        return model_info
        
    except Exception as e:
        logger.error(f"Error fetching models for {model_type}: {e}")
        return []

def to_valid_identifier(text):
    """Convert text to a valid Python identifier."""
    # Replace hyphens with underscores
    text = text.replace("-", "_")
    # Remove any other invalid characters
    text = re.sub(r'[^a-zA-Z0-9_]', '', text)
    # Ensure it doesn't start with a number
    if text and text[0].isdigit():
        text = '_' + text
    return text

def get_architecture_type(model_type):
    """Determine architecture type based on model type."""
    model_type_lower = model_type.lower()
    for arch_type, models in ARCHITECTURE_TYPES.items():
        if any(model in model_type_lower for model in models):
            return arch_type
    return "unknown"  # Default if unknown

def get_class_name_capitalization(model_type):
    """Get the proper capitalization for a model type."""
    # First check for exact matches
    if model_type.lower() in CLASS_NAME_CAPITALIZATION:
        return CLASS_NAME_CAPITALIZATION[model_type.lower()]
    
    # Then check for partial matches
    for prefix, capitalization in CLASS_NAME_CAPITALIZATION.items():
        if model_type.lower().startswith(prefix.lower()):
            # Handle case where model_type is like "gpt-j-123" or "bert-base"
            suffix = model_type[len(prefix):]
            return capitalization + suffix
    
    # Default to capitalizing first letter of each part
    parts = model_type.split('-')
    return ''.join(part.capitalize() for part in parts)

def find_models_in_transformers():
    """Find all model classes in the transformers library."""
    try:
        import transformers
        models = []
        
        # Collect all model classes from the transformers package
        for name, obj in inspect.getmembers(transformers):
            # Look for classes that represent models
            if inspect.isclass(obj) and 'Model' in name and obj.__module__.startswith('transformers.'):
                model_type = None
                
                # Try to extract the model type from the module name
                module_parts = obj.__module__.split('.')
                if len(module_parts) >= 3:
                    model_type = module_parts[2]  # e.g., transformers.models.bert
                    
                    # Skip auto models
                    if model_type == 'auto':
                        continue
                        
                    # Fix model_type if it's a submodule
                    if model_type in ['configuration', 'modeling']:
                        # Get the model type from a deeper level
                        if len(module_parts) >= 4:
                            model_type = module_parts[3]
                    
                # Add the model to our list
                if model_type:
                    models.append({
                        'name': name,
                        'type': model_type,
                        'module': obj.__module__,
                        'is_hyphenated': '-' in model_type,
                        'architecture': get_architecture_type(model_type)
                    })
                
        return models
    except ImportError:
        logger.warning("Transformers library not available. Using static model list.")
        return []

def find_hyphenated_models():
    """Find models with hyphenated names that need special handling."""
    # Start with a list of known hyphenated models
    known_hyphenated_models = [
        {"type": "gpt-j", "name": "GPTJForCausalLM", "architecture": "decoder-only"},
        {"type": "gpt-neo", "name": "GPTNeoForCausalLM", "architecture": "decoder-only"},
        {"type": "gpt-neox", "name": "GPTNeoXForCausalLM", "architecture": "decoder-only"},
        {"type": "xlm-roberta", "name": "XLMRobertaForMaskedLM", "architecture": "encoder-only"}
    ]
    
    # Try to find models from the transformers library
    transformers_models = find_models_in_transformers()
    
    # Combine the lists, prioritizing transformers models
    combined_models = []
    for model in transformers_models:
        if model['is_hyphenated']:
            combined_models.append(model)
    
    # Add any known hyphenated models that weren't found
    known_types = {model['type'] for model in combined_models}
    for model in known_hyphenated_models:
        if model['type'] not in known_types:
            combined_models.append(model)
    
    return combined_models

def get_recommended_default_model(model_type):
    """Query the HuggingFace API to find the most suitable default model for a model type."""
    # Query HuggingFace API for popular models
    popular_models = query_huggingface_api(model_type, limit=5)
    
    if not popular_models:
        # If API query fails, use fallback defaults
        fallback_defaults = {
            "bert": "bert-base-uncased",
            "gpt2": "gpt2",
            "t5": "t5-small",
            "vit": "google/vit-base-patch16-224",
            "gpt-j": "EleutherAI/gpt-j-6b",
            "gpt-neo": "EleutherAI/gpt-neo-1.3B",
            "xlm-roberta": "xlm-roberta-base",
            "whisper": "openai/whisper-tiny",
            "wav2vec2": "facebook/wav2vec2-base-960h",
            "clip": "openai/clip-vit-base-patch32"
        }
        return fallback_defaults.get(model_type, f"{model_type}-base")
    
    # Prefer smaller models that are still popular
    # First look for models with "base" or "small" in the name
    base_or_small_models = [m for m in popular_models if "base" in m["id"].lower() or "small" in m["id"].lower()]
    if base_or_small_models:
        return base_or_small_models[0]["id"]
    
    # If no base/small models, use the most popular one
    return popular_models[0]["id"]

def generate_registry_entry(model_data, query_default=False):
    """Generate a MODEL_REGISTRY entry for a model."""
    model_type = model_data['type']
    model_name = model_data['name']
    model_architecture = model_data['architecture']
    
    # Convert model type to valid Python identifier
    model_type_identifier = to_valid_identifier(model_type)
    
    # Get capitalized version for family name
    if '-' in model_type:
        family_name = get_class_name_capitalization(model_type)
    else:
        family_name = model_type.upper() if model_type.lower() == model_type else model_type
    
    # Determine test class name
    if '-' in model_type:
        # Handle hyphenated models specially
        model_capitalized = ''.join(part.capitalize() for part in model_type.split('-'))
        test_class = f"Test{model_capitalized}Models"
    else:
        test_class = f"Test{model_type.capitalize()}Models"
    
    # Determine module name
    module_name = f"test_hf_{model_type_identifier}"
    
    # Determine default task based on architecture
    task = "fill-mask"
    if model_architecture == "decoder-only":
        task = "text-generation"
    elif model_architecture == "encoder-decoder":
        task = "text2text-generation"
    elif model_architecture == "vision":
        task = "image-classification"
    elif model_architecture == "speech":
        task = "automatic-speech-recognition"
    elif model_architecture == "multimodal" or model_architecture == "vision-text":
        task = "image-to-text"
    
    # Generate placeholder input
    input_text = f"{model_type} is a "
    if task == "fill-mask":
        input_text += "[MASK] model."
    elif task == "text-generation":
        input_text += "transformer model that"
    
    # Get default model - either from API or use fallback
    if query_default:
        default_model = get_recommended_default_model(model_type)
        logger.info(f"Using recommended model for {model_type}: {default_model}")
    else:
        # Use heuristic approach
        default_model = f"{model_type}-base" if '-base' not in model_type else model_type
    
    # Create the registry entry
    registry_entry = {
        "family_name": family_name,
        "description": f"{family_name} models",
        "default_model": default_model,
        "class": model_name,
        "test_class": test_class,
        "module_name": module_name,
        "tasks": [task],
        "inputs": {
            "text": input_text
        }
    }
    
    # Add task-specific args for text generation
    if task == "text-generation":
        registry_entry["task_specific_args"] = {
            "text-generation": {
                "max_length": 50
            }
        }
    
    return registry_entry

def format_registry_entry(model_type, entry):
    """Format a registry entry as Python code for MODEL_REGISTRY."""
    lines = [f'    "{model_type}": {{']
    for key, value in entry.items():
        if isinstance(value, str):
            lines.append(f'        "{key}": "{value}",')
        elif isinstance(value, list):
            lines.append(f'        "{key}": {value},')
        elif isinstance(value, dict):
            lines.append(f'        "{key}": {{')
            for k, v in value.items():
                if isinstance(v, dict):
                    lines.append(f'            "{k}": {{')
                    for inner_k, inner_v in v.items():
                        lines.append(f'                "{inner_k}": {inner_v},')
                    lines.append('            },')
                else:
                    lines.append(f'            "{k}": "{v}",')
            lines.append('        },')
    lines.append('    },')
    return '\n'.join(lines)

def format_class_name_fix(model_name, correct_name):
    """Format a class name fix entry for CLASS_NAME_FIXES."""
    return f'    "{model_name}": "{correct_name}",'

def save_model_registry(registry_data, output_path):
    """Save the model registry data to a JSON file."""
    try:
        with open(output_path, 'w') as f:
            json.dump(registry_data, f, indent=2)
        logger.info(f"Saved model registry to {output_path}")
        return True
    except Exception as e:
        logger.error(f"Error saving model registry: {e}")
        return False

def main():
    """Main function to find models and generate entries."""
    parser = argparse.ArgumentParser(description="Find and categorize models from HuggingFace transformers")
    parser.add_argument("--hyphenated-only", action="store_true", help="Only find hyphenated model names")
    parser.add_argument("--max-models", type=int, default=None, help="Maximum number of models to display")
    parser.add_argument("--output", type=str, help="Output file for model registry entries")
    parser.add_argument("--query-huggingface", action="store_true", help="Query HuggingFace API for default models")
    parser.add_argument("--update-registry", action="store_true", help="Update the model registry file")
    parser.add_argument("--model-type", type=str, help="Query a specific model type")
    parser.add_argument("--verify", type=str, help="Verify a specific model name")
    parser.add_argument("--registry-file", type=str, default="huggingface_model_types.json", help="Path to save/load the registry file")
    
    args = parser.parse_args()
    
    # Single model verification
    if args.verify:
        model_name = args.verify
        print(f"Verifying model: {model_name}")
        result = try_model_access(model_name)
        
        if result.get("error"):
            print(f"Error: {result['error']}")
        else:
            print(f"Model exists: {result['id']}")
            print(f"Tags: {', '.join(result.get('tags', []))}")
        return 0
    
    # Single model type query
    if args.model_type:
        model_type = args.model_type
        print(f"Querying models for type: {model_type}")
        
        popular_models = query_huggingface_api(model_type, limit=10)
        if popular_models:
            print(f"\nFound {len(popular_models)} models for {model_type}:")
            for i, model in enumerate(popular_models, 1):
                downloads = f"{model.get('downloads', 0):,}" if model.get('downloads') else "Unknown"
                print(f"{i}. {model['id']} (Downloads: {downloads})")
            
            # Recommend default model
            default_model = get_recommended_default_model(model_type)
            print(f"\nRecommended default model: {default_model}")
            
            # Save to registry if requested
            if args.update_registry:
                registry_data = {}
                
                # Try to load existing registry first
                if os.path.exists(args.registry_file):
                    try:
                        with open(args.registry_file, 'r') as f:
                            registry_data = json.load(f)
                            # Ensure registry_data is a dictionary
                            if not isinstance(registry_data, dict):
                                logger.warning(f"Registry file contains invalid data, initializing empty dictionary")
                                registry_data = {}
                    except Exception as e:
                        logger.error(f"Error loading registry file: {e}")
                
                # Update with new model info
                registry_data[model_type] = {
                    "default_model": default_model,
                    "models": [m["id"] for m in popular_models],
                    "downloads": {m["id"]: m.get("downloads", 0) for m in popular_models},
                    "updated_at": datetime.now().isoformat()
                }
                
                # Save updated registry
                save_model_registry(registry_data, args.registry_file)
        else:
            print(f"No models found for {model_type}")
        
        return 0
    
    # Find models from transformers library
    if args.hyphenated_only:
        logger.info("Finding hyphenated model names in transformers...")
        models = find_hyphenated_models()
    else:
        logger.info("Finding all model architectures in transformers...")
        models = find_models_in_transformers()
    
    # Limit number of models if requested
    if args.max_models and len(models) > args.max_models:
        models = models[:args.max_models]
    
    # Generate registry entries
    registry_entries = {}
    for model in models:
        model_type = model['type']
        registry_entries[model_type] = generate_registry_entry(model, query_default=args.query_huggingface)
    
    # Print summary
    print(f"\nFound {len(models)} models in transformers library")
    print("\nModel architectures summary:")
    
    # Group by architecture
    by_architecture = {}
    for model in models:
        arch = model['architecture']
        if arch not in by_architecture:
            by_architecture[arch] = []
        by_architecture[arch].append(model)
    
    for arch, models_in_arch in sorted(by_architecture.items()):
        print(f"- {arch}: {len(models_in_arch)} models")
        for model in sorted(models_in_arch, key=lambda x: x['type']):
            print(f"  - {model['type']}: {model['name']}")
    
    # Identify hyphenated models
    hyphenated_models = [model for model in models if '-' in model['type']]
    print(f"\nFound {len(hyphenated_models)} hyphenated model names:")
    for model in sorted(hyphenated_models, key=lambda x: x['type']):
        print(f"- {model['type']}: {model['name']}")
    
    if args.query_huggingface:
        print("\nDefault models from HuggingFace API:")
        for model_type, entry in sorted(registry_entries.items()):
            print(f"  - {model_type}: {entry['default_model']}")
    
    # Generate MODEL_REGISTRY entries
    print("\nModel registry entries:")
    for model_type, entry in sorted(registry_entries.items()):
        formatted_entry = format_registry_entry(model_type, entry)
        print(formatted_entry)
    
    # Generate CLASS_NAME_FIXES entries
    print("\nClass name fixes:")
    for model in models:
        model_type = model['type']
        if '-' in model_type:
            # Create all possible capitalization patterns
            parts = model_type.split('-')
            incorrect_name = ''.join(part.capitalize() for part in parts)
            correct_capitalization = get_class_name_capitalization(model_type)
            
            # Extract the model class part
            if "ForCausalLM" in model['name']:
                correct_name = f"{correct_capitalization}ForCausalLM"
                print(format_class_name_fix(f"{incorrect_name}ForCausalLM", correct_name))
            elif "Model" in model['name']:
                # Handle general Model classes
                for suffix in ["Model", "ForMaskedLM", "ForSequenceClassification", "ForTokenClassification", 
                               "ForQuestionAnswering", "ForMultipleChoice", "ForImageClassification",
                               "ForAudioClassification", "ForCTC"]:
                    if suffix in model['name']:
                        correct_name = f"{correct_capitalization}{suffix}"
                        print(format_class_name_fix(f"{incorrect_name}{suffix}", correct_name))
    
    # Save to file if requested
    if args.output:
        output_data = {
            "model_registry_entries": registry_entries,
            "class_name_fixes": {model['type']: get_class_name_capitalization(model['type']) 
                               for model in hyphenated_models},
            "models_found": [model['type'] for model in models],
            "hyphenated_models": [model['type'] for model in hyphenated_models],
            "generated_at": datetime.now().isoformat()
        }
        
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        logger.info(f"Saved output to {args.output}")
    
    # Update registry file if requested
    if args.update_registry and not args.model_type:
        registry_data = {}
        
        # Convert entries to registry format
        for model_type, entry in registry_entries.items():
            registry_data[model_type] = {
                "default_model": entry["default_model"],
                "description": entry["description"],
                "class": entry["class"],
                "architecture": next((model["architecture"] for model in models if model["type"] == model_type), "unknown"),
                "updated_at": datetime.now().isoformat()
            }
        
        # Save the registry
        save_model_registry(registry_data, args.registry_file)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())