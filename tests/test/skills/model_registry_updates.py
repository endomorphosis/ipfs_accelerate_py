#!/usr/bin/env python3

"""
Script to add missing high-priority models to the MODEL_REGISTRY in test_generator_fixed.py
based on the HF_MODEL_COVERAGE_ROADMAP.md requirements.

This script will:
1. Add new model entries to the MODEL_REGISTRY
2. Update the ARCHITECTURE_TYPES dictionary if needed
3. Print a report of changes that will be made
"""

import os
import re
import sys
import shutil
import argparse
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define new model entries to add to the registry
NEW_MODELS = {
    "qwen2": {
        "family_name": "Qwen2",
        "description": "Qwen2 large language model developed by Alibaba",
        "default_model": "Qwen/Qwen2-7B",
        "class": "Qwen2ForCausalLM",
        "test_class": "TestQwen2Models",
        "module_name": "test_hf_qwen2",
        "tasks": ['text-generation'],
        "inputs": {
            "text": "Qwen2 is a large language model that",
        },
    },
    "qwen3": {
        "family_name": "Qwen3",
        "description": "Qwen3 large language model developed by Alibaba",
        "default_model": "Qwen/Qwen3-7B",
        "class": "Qwen3ForCausalLM",
        "test_class": "TestQwen3Models",
        "module_name": "test_hf_qwen3",
        "tasks": ['text-generation'],
        "inputs": {
            "text": "Qwen3 is a large language model that",
        },
    },
    "codellama": {
        "family_name": "CodeLLaMA",
        "description": "CodeLLaMA is a code-specialized version of LLaMA",
        "default_model": "codellama/CodeLlama-7b-hf",
        "class": "LlamaForCausalLM",
        "test_class": "TestCodeLlamaModels",
        "module_name": "test_hf_codellama",
        "tasks": ['text-generation', 'code-generation'],
        "inputs": {
            "text": "def fibonacci(n):",
        },
    },
    "fuyu": {
        "family_name": "Fuyu",
        "description": "Fuyu multimodal model by Adept",
        "default_model": "adept/fuyu-8b",
        "class": "FuyuForCausalLM",
        "test_class": "TestFuyuModels",
        "module_name": "test_hf_fuyu",
        "tasks": ['visual-question-answering'],
        "inputs": {
            "text": "What is shown in this image?",
        },
    },
    "kosmos-2": {
        "family_name": "Kosmos-2",
        "description": "Kosmos-2 multimodal model with grounding capabilities",
        "default_model": "microsoft/kosmos-2-patch14-224",
        "class": "Kosmos2ForConditionalGeneration",
        "test_class": "TestKosmos2Models",
        "module_name": "test_hf_kosmos2",
        "tasks": ['visual-question-answering', 'image-to-text', 'image-grounding'],
        "inputs": {
            "text": "What is shown in this image?",
        },
    },
    "llava-next": {
        "family_name": "LLaVA-Next",
        "description": "Next generation of LLaVA with improved capabilities",
        "default_model": "llava-hf/llava-v1.6-mistral-7b-hf",
        "class": "LlavaNextForConditionalGeneration",
        "test_class": "TestLlavaNextModels",
        "module_name": "test_hf_llava_next",
        "tasks": ['visual-question-answering'],
        "inputs": {
            "text": "What is shown in this image?",
        },
    },
    "video-llava": {
        "family_name": "Video-LLaVA",
        "description": "LLaVA model extended for video understanding",
        "default_model": "LanguageBind/Video-LLaVA-7B-hf",
        "class": "VideoLlavaForConditionalGeneration",
        "test_class": "TestVideoLlavaModels",
        "module_name": "test_hf_video_llava",
        "tasks": ['video-question-answering'],
        "inputs": {
            "text": "What is happening in this video?",
        },
    },
    "bark": {
        "family_name": "Bark",
        "description": "Text-to-audio model by Suno",
        "default_model": "suno/bark-small",
        "class": "BarkModel",
        "test_class": "TestBarkModels",
        "module_name": "test_hf_bark",
        "tasks": ['text-to-audio'],
        "inputs": {
            "text": "Hello, my name is Suno. And, I like to sing.",
        },
    },
    "mobilenet-v2": {
        "family_name": "MobileNetV2",
        "description": "Lightweight vision model optimized for mobile and edge devices",
        "default_model": "google/mobilenet_v2_1.0_224",
        "class": "MobileNetV2ForImageClassification",
        "test_class": "TestMobileNetV2Models",
        "module_name": "test_hf_mobilenet_v2",
        "tasks": ['image-classification'],
        "inputs": {},
    },
    "blip-2": {
        "family_name": "BLIP-2",
        "description": "BLIP-2 vision-language model with improved architecture",
        "default_model": "Salesforce/blip2-opt-2.7b",
        "class": "Blip2ForConditionalGeneration",
        "test_class": "TestBlip2Models",
        "module_name": "test_hf_blip_2",
        "tasks": ['image-to-text', 'visual-question-answering'],
        "inputs": {
            "text": "What is shown in this image?",
        },
    },
    "chinese-clip": {
        "family_name": "ChineseCLIP",
        "description": "Chinese CLIP model for vision-text understanding",
        "default_model": "OFA-Sys/chinese-clip-vit-base-patch16",
        "class": "ChineseCLIPModel",
        "test_class": "TestChineseCLIPModels",
        "module_name": "test_hf_chinese_clip",
        "tasks": ['zero-shot-image-classification'],
        "inputs": {},
    },
    "clipseg": {
        "family_name": "CLIPSeg",
        "description": "CLIP with segmentation capabilities",
        "default_model": "CIDAS/clipseg-rd64-refined",
        "class": "CLIPSegForImageSegmentation",
        "test_class": "TestCLIPSegModels",
        "module_name": "test_hf_clipseg",
        "tasks": ['image-segmentation'],
        "inputs": {
            "text": "person",
        },
    },
}

# Architecture type updates
ARCHITECTURE_UPDATES = {
    "decoder-only": ["qwen2", "qwen3", "codellama"],
    "multimodal": ["llava-next", "video-llava", "fuyu", "kosmos-2"],
    "vision": ["mobilenet-v2"],
    "vision-text": ["blip-2", "chinese-clip", "clipseg"],
    "speech": ["bark"]
}

def update_model_registry(generator_file, apply_changes=False):
    """
    Update the MODEL_REGISTRY in test_generator_fixed.py with new model entries.
    
    Args:
        generator_file (str): Path to the test_generator_fixed.py file
        apply_changes (bool): If True, write changes to the file
        
    Returns:
        bool: True if successful, False otherwise
    """
    logger.info(f"Updating MODEL_REGISTRY in {generator_file}")
    
    # Create backup file
    backup_file = f"{generator_file}.bak"
    shutil.copy2(generator_file, backup_file)
    logger.info(f"Created backup at {backup_file}")
    
    try:
        # Read the file content
        with open(generator_file, 'r') as f:
            content = f.read()
        
        # Find the MODEL_REGISTRY start
        registry_match = re.search(r'MODEL_REGISTRY = \{', content)
        if not registry_match:
            logger.error("Couldn't find MODEL_REGISTRY in the file")
            return False
        
        # Find the end of the registry (look for closing brace followed by a function or comment)
        registry_end_match = re.search(r'\}(\s*\n\s*)(def|if|#)', content[registry_match.start():])
        if not registry_end_match:
            logger.error("Couldn't find the end of MODEL_REGISTRY")
            return False
        
        # Extract the current registry content
        registry_start = registry_match.start()
        registry_end = registry_start + registry_end_match.start() + 1  # +1 to include the closing brace
        
        registry_content = content[registry_start:registry_end]
        
        # Check which models already exist in the registry
        existing_models = []
        for model_key in NEW_MODELS.keys():
            if f'"{model_key}":' in registry_content:
                existing_models.append(model_key)
                logger.info(f"Model '{model_key}' already exists in the registry")
        
        # Prepare new model entries for models that don't exist yet
        new_entries = []
        for model_key, model_data in NEW_MODELS.items():
            if model_key not in existing_models:
                entry = f'    "{model_key}": {{\n'
                entry += f'        "family_name": "{model_data["family_name"]}",\n'
                entry += f'        "description": "{model_data["description"]}",\n'
                entry += f'        "default_model": "{model_data["default_model"]}",\n'
                entry += f'        "class": "{model_data["class"]}",\n'
                entry += f'        "test_class": "{model_data["test_class"]}",\n'
                entry += f'        "module_name": "{model_data["module_name"]}",\n'
                entry += f'        "tasks": {repr(model_data["tasks"])},\n'
                entry += f'        "inputs": {{\n'
                for input_key, input_value in model_data.get("inputs", {}).items():
                    entry += f'            "{input_key}": "{input_value}",\n'
                entry += f'        }},\n'
                if "task_specific_args" in model_data:
                    entry += f'        "task_specific_args": {{\n'
                    for task, args in model_data["task_specific_args"].items():
                        entry += f'            "{task}": {{\n'
                        for arg_key, arg_value in args.items():
                            entry += f'                "{arg_key}": {arg_value},\n'
                        entry += f'            }},\n'
                    entry += f'        }},\n'
                entry += f'    }},\n'
                new_entries.append(entry)
        
        # Build the updated registry content
        if new_entries:
            # Insert new entries before the closing brace
            new_registry_content = registry_content[:-1] + '\n' + ''.join(new_entries) + '}'
            updated_content = content[:registry_start] + new_registry_content + content[registry_end:]
        else:
            updated_content = content
        
        # Look for ARCHITECTURE_TYPES dictionary
        arch_types_match = re.search(r'ARCHITECTURE_TYPES\s*=\s*\{', content)
        if arch_types_match:
            for arch_type, models in ARCHITECTURE_UPDATES.items():
                # For each architecture type, find its line in the content
                arch_pattern = re.compile(fr'(\s*"{arch_type}":\s*\[)([^\]]+)(\],)', re.MULTILINE)
                arch_match = arch_pattern.search(content)
                
                if arch_match:
                    prefix = arch_match.group(1)
                    current_models = arch_match.group(2)
                    suffix = arch_match.group(3)
                    
                    # Parse current models and determine which ones need to be added
                    current_model_list = [m.strip(' "\'') for m in current_models.split(',')]
                    models_to_add = [m for m in models if m not in current_model_list]
                    
                    if models_to_add:
                        # Add the new models to the list
                        if current_models.strip():
                            new_models_str = current_models.rstrip() + ', "' + '", "'.join(models_to_add) + '"'
                        else:
                            new_models_str = '"' + '", "'.join(models_to_add) + '"'
                        
                        # Create updated line
                        new_arch_line = prefix + new_models_str + suffix
                        
                        # Replace in the updated content
                        updated_content = updated_content.replace(arch_match.group(0), new_arch_line)
                        
                        logger.info(f"Added {models_to_add} to {arch_type} in ARCHITECTURE_TYPES")
        
        # Print a summary of changes
        print("\n=== Summary of Changes ===")
        print(f"New models to be added: {len(new_entries)}")
        for model_key in NEW_MODELS:
            if model_key not in existing_models:
                print(f"  - {model_key} ({NEW_MODELS[model_key]['family_name']})")
        
        print("\nArchitecture types to be updated:")
        for arch_type, models in ARCHITECTURE_UPDATES.items():
            models_to_add = []
            if arch_types_match:
                arch_pattern = re.compile(fr'(\s*"{arch_type}":\s*\[)([^\]]+)(\],)', re.MULTILINE)
                arch_match = arch_pattern.search(content)
                if arch_match:
                    current_models = arch_match.group(2)
                    current_model_list = [m.strip(' "\'') for m in current_models.split(',')]
                    models_to_add = [m for m in models if m not in current_model_list]
            print(f"  - {arch_type}: Adding {models_to_add}")
        
        # Write the updated content back to the file if requested
        if apply_changes:
            with open(generator_file, 'w') as f:
                f.write(updated_content)
            logger.info(f"Applied changes to {generator_file}")
        else:
            logger.info("Dry run - no changes written to file")
        
        return True
        
    except Exception as e:
        logger.error(f"Error updating MODEL_REGISTRY: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    parser = argparse.ArgumentParser(description="Update MODEL_REGISTRY with new model entries")
    parser.add_argument("--apply", action="store_true", help="Apply changes (without this flag, runs in dry-run mode)")
    parser.add_argument("--generator-file", type=str, 
                      default=os.path.join(os.path.dirname(__file__), "test_generator_fixed.py"),
                      help="Path to test_generator_fixed.py")
    
    args = parser.parse_args()
    
    # Verify the generator file exists
    if not os.path.exists(args.generator_file):
        logger.error(f"Generator file not found: {args.generator_file}")
        return 1
    
    # Update the model registry
    success = update_model_registry(args.generator_file, apply_changes=args.apply)
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())