#!/usr/bin/env python3
"""
Regenerate test files for models with hyphenated names using templates.

This script:
1. Identifies test files with hyphenated model names
2. Determines the appropriate template for each
3. Recreates the test file using the template with proper naming

Usage:
    python regenerate_hyphenated_tests.py
"""

import os
import sys
import re
import argparse
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

# Map model types to architecture types
ARCHITECTURE_TYPES = {
    "encoder-only": ["bert", "distilbert", "roberta", "electra", "albert", "camembert", "xlm-roberta", "deberta"],
    "decoder-only": ["gpt2", "gpt-j", "gptj", "gpt-neo", "gpt_neo", "gpt_neox", "bloom", "llama", "mistral", "falcon", "phi", "mixtral", "mpt", "opt"],
    "encoder-decoder": ["t5", "bart", "pegasus", "mbart", "mt5", "longt5", "led", "marian"],
    "vision": ["vit", "swin", "deit", "beit", "convnext", "poolformer", "dinov2"],
    "vision-text": ["clip", "blip"],
    "speech": ["wav2vec2", "hubert", "whisper", "bark", "speecht5"],
    "multimodal": ["llava", "clip", "blip", "git", "pix2struct", "paligemma", "video-llava"]
}

# Special cases for specific files
SPECIAL_CASES = {
    "encoder-only": "encoder_only",
    "decoder-only": "decoder_only",
    "encoder-decoder": "encoder_decoder"
}

def to_valid_identifier(text):
    """Convert text to a valid Python identifier."""
    # Check if it's a special case first
    if text in SPECIAL_CASES:
        return SPECIAL_CASES[text]
        
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
    # Special cases first
    if model_type == "encoder-only":
        return "encoder-only"
    elif model_type == "decoder-only":
        return "decoder-only"
    elif model_type == "encoder-decoder":
        return "encoder-decoder"
        
    model_type_lower = model_type.lower()
    for arch_type, models in ARCHITECTURE_TYPES.items():
        if any(model in model_type_lower for model in models):
            return arch_type
    return "encoder-only"  # Default to encoder-only if unknown

def get_template_for_architecture(arch_type):
    """Get the template path for a specific architecture type."""
    # Define base directory for templates
    template_dir = os.path.join(os.path.dirname(__file__), "templates")
    
    template_map = {
        "encoder-only": os.path.join(template_dir, "encoder_only_template.py"),
        "decoder-only": os.path.join(template_dir, "decoder_only_template.py"),
        "encoder-decoder": os.path.join(template_dir, "encoder_decoder_template.py"),
        "vision": os.path.join(template_dir, "vision_template.py"),
        "vision-text": os.path.join(template_dir, "vision_text_template.py"),
        "speech": os.path.join(template_dir, "speech_template.py"),
        "multimodal": os.path.join(template_dir, "multimodal_template.py")
    }
    
    template_path = template_map.get(arch_type)
    if not template_path or not os.path.exists(template_path):
        logger.warning(f"Template not found for {arch_type}, using encoder-only template")
        fallback_template = os.path.join(template_dir, "encoder_only_template.py")
        if not os.path.exists(fallback_template):
            logger.error(f"Fallback template not found: {fallback_template}")
            return None
        return fallback_template
        
    return template_path

def get_default_model_for_type(model_type):
    """Get default model ID for a model type."""
    default_models = {
        "bert": "bert-base-uncased",
        "roberta": "roberta-base",
        "distilbert": "distilbert-base-uncased",
        "electra": "google/electra-small-discriminator",
        "albert": "albert-base-v2",
        "xlm-roberta": "xlm-roberta-base",
        "gpt2": "gpt2",
        "gptj": "EleutherAI/gpt-j-6b",
        "gpt-j": "EleutherAI/gpt-j-6b",
        "gpt-neo": "EleutherAI/gpt-neo-125m",
        "gpt_neo": "EleutherAI/gpt-neo-125m",
        "gpt_neox": "EleutherAI/gpt-neox-20b",
        "bloom": "bigscience/bloom-560m",
        "llama": "meta-llama/Llama-2-7b",
        "opt": "facebook/opt-125m",
        "t5": "t5-small",
        "bart": "facebook/bart-base",
        "pegasus": "google/pegasus-xsum",
        "mbart": "facebook/mbart-large-cc25",
        "mt5": "google/mt5-small",
        "vit": "google/vit-base-patch16-224",
        "swin": "microsoft/swin-tiny-patch4-window7-224",
        "deit": "facebook/deit-base-patch16-224",
        "beit": "microsoft/beit-base-patch16-224",
        "convnext": "facebook/convnext-tiny-224",
        "clip": "openai/clip-vit-base-patch32",
        "blip": "Salesforce/blip-vqa-base",
        "llava": "llava-hf/llava-1.5-7b-hf",
        "wav2vec2": "facebook/wav2vec2-base-960h",
        "hubert": "facebook/hubert-base-ls960",
        "whisper": "openai/whisper-small",
        "encoder-only": "bert-base-uncased",
        "decoder-only": "gpt2",
        "encoder-decoder": "t5-small"
    }
    
    # Return the default model if found, otherwise construct a reasonable default
    model_type_lower = model_type.lower()
    if model_type_lower in default_models:
        return default_models[model_type_lower]
        
    # Try to find a close match
    for key in default_models:
        if key in model_type_lower:
            return default_models[key]
            
    # Fallback to a generic model name
    return f"{model_type.replace('-', '_')}-base"

def regenerate_test_file(model_type, output_dir="fixed_tests"):
    """
    Regenerate a test file for a specific model type using templates.
    
    Args:
        model_type: Type of model (e.g., bert, gpt2, t5)
        output_dir: Directory to save the test file
        
    Returns:
        Tuple of (success, output_path)
    """
    try:
        # Determine output path - we keep the hyphenated filename
        output_path = os.path.join(output_dir, f"test_hf_{model_type}.py")
        
        # Determine architecture type
        arch_type = get_architecture_type(model_type)
        logger.info(f"Model {model_type} has architecture type: {arch_type}")
        
        # Get template file
        template_path = get_template_for_architecture(arch_type)
        if not template_path:
            logger.error(f"No template found for architecture type: {arch_type}")
            return False, None
        
        # Get default model for this type
        default_model = get_default_model_for_type(model_type)
        
        # Read template
        with open(template_path, 'r') as f:
            template_content = f.read()
        
        # Create valid Python identifiers
        model_valid = to_valid_identifier(model_type)
        model_upper = model_valid.upper()
        
        # Create a capitalized version for class names
        if '-' in model_type:
            model_capitalized = ''.join(part.capitalize() for part in model_type.split('-'))
        else:
            model_capitalized = model_type.capitalize()
        
        content = template_content
        
        # Replace based on architecture type
        if arch_type == "encoder-only":
            replacements = {
                "BERT_MODELS_REGISTRY": f"{model_upper}_MODELS_REGISTRY",
                "TestBertModels": f"Test{model_capitalized}Models",
                "bert": model_valid,  # Use valid identifier here to avoid issues
                "BertForMaskedLM": f"{model_capitalized}ForMaskedLM",
                "bert-base-uncased": default_model,
                "fill-mask": "fill-mask",
                "hf_bert_": f"hf_{model_valid}_",
                "bert_tester": f"{model_valid}_tester",
                "BERT model": f"{model_capitalized} model",
                "BERT": model_capitalized
            }
        elif arch_type == "decoder-only":
            replacements = {
                "GPT2_MODELS_REGISTRY": f"{model_upper}_MODELS_REGISTRY",
                "TestGpt2Models": f"Test{model_capitalized}Models",
                "gpt2": model_valid,  # Use valid identifier here to avoid issues
                "GPT2LMHeadModel": f"{model_capitalized}LMHeadModel",
                "gpt2-medium": default_model,
                "text-generation": "text-generation",
                "hf_gpt2_": f"hf_{model_valid}_",
                "gpt2_tester": f"{model_valid}_tester",
                "GPT2 model": f"{model_capitalized} model",
                "GPT2": model_capitalized
            }
        elif arch_type == "encoder-decoder":
            replacements = {
                "T5_MODELS_REGISTRY": f"{model_upper}_MODELS_REGISTRY",
                "TestT5Models": f"Test{model_capitalized}Models",
                "t5": model_valid,  # Use valid identifier here to avoid issues
                "T5ForConditionalGeneration": f"{model_capitalized}ForConditionalGeneration",
                "t5-small": default_model,
                "text2text-generation": "text2text-generation",
                "hf_t5_": f"hf_{model_valid}_",
                "t5_tester": f"{model_valid}_tester",
                "T5 model": f"{model_capitalized} model",
                "T5": model_capitalized
            }
        elif arch_type == "vision":
            replacements = {
                "VIT_MODELS_REGISTRY": f"{model_upper}_MODELS_REGISTRY",
                "TestVitModels": f"Test{model_capitalized}Models",
                "vit": model_valid,  # Use valid identifier here to avoid issues
                "ViTForImageClassification": f"{model_capitalized}ForImageClassification",
                "google/vit-base-patch16-224": default_model,
                "image-classification": "image-classification",
                "hf_vit_": f"hf_{model_valid}_",
                "vit_tester": f"{model_valid}_tester",
                "ViT model": f"{model_capitalized} model",
                "ViT": model_capitalized
            }
        elif arch_type in ["vision-text", "multimodal"]:
            replacements = {
                "CLIP_MODELS_REGISTRY": f"{model_upper}_MODELS_REGISTRY",
                "TestClipModels": f"Test{model_capitalized}Models",
                "clip": model_valid,  # Use valid identifier here to avoid issues
                "CLIPModel": f"{model_capitalized}Model",
                "openai/clip-vit-base-patch32": default_model,
                "image-to-text": "image-to-text",
                "hf_clip_": f"hf_{model_valid}_",
                "clip_tester": f"{model_valid}_tester",
                "CLIP model": f"{model_capitalized} model",
                "CLIP": model_capitalized
            }
        elif arch_type == "speech":
            replacements = {
                "WHISPER_MODELS_REGISTRY": f"{model_upper}_MODELS_REGISTRY",
                "TestWhisperModels": f"Test{model_capitalized}Models",
                "whisper": model_valid,  # Use valid identifier here to avoid issues
                "WhisperForConditionalGeneration": f"{model_capitalized}ForConditionalGeneration",
                "openai/whisper-small": default_model,
                "automatic-speech-recognition": "automatic-speech-recognition",
                "hf_whisper_": f"hf_{model_valid}_",
                "whisper_tester": f"{model_valid}_tester",
                "Whisper model": f"{model_capitalized} model",
                "Whisper": model_capitalized
            }
        else:
            replacements = {
                "BERT_MODELS_REGISTRY": f"{model_upper}_MODELS_REGISTRY",
                "TestBertModels": f"Test{model_capitalized}Models",
                "bert": model_valid,  # Use valid identifier here to avoid issues
                "bert-base-uncased": default_model,
                "hf_bert_": f"hf_{model_valid}_",
                "bert_tester": f"{model_valid}_tester",
                "BERT model": f"{model_capitalized} model",
                "BERT": model_capitalized
            }
        
        for old, new in replacements.items():
            content = content.replace(old, new)
        
        # Create directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Make a backup of the original file if it exists
        if os.path.exists(output_path):
            backup_path = f"{output_path}.regenerate.bak"
            with open(output_path, 'r') as f:
                original = f.read()
            with open(backup_path, 'w') as f:
                f.write(original)
            logger.info(f"Created backup of original file at: {backup_path}")
        
        # Write the regenerated file
        with open(output_path, 'w') as f:
            f.write(content)
        
        logger.info(f"Regenerated test file: {output_path}")
        
        # Verify syntax
        try:
            compile(content, output_path, 'exec')
            logger.info(f"✅ {output_path}: Syntax is valid")
            return True, output_path
        except SyntaxError as e:
            logger.error(f"❌ {output_path}: Syntax error: {e}")
            return False, output_path
            
    except Exception as e:
        logger.error(f"Error regenerating test file for {model_type}: {e}")
        return False, None

def main():
    parser = argparse.ArgumentParser(description="Regenerate test files for models with hyphenated names")
    parser.add_argument("--models", type=str, nargs="+", help="Specific model types to regenerate")
    parser.add_argument("--find-hyphenated", action="store_true", help="Find and fix all hyphenated model test files")
    parser.add_argument("--output-dir", type=str, default="fixed_tests", help="Directory to save regenerated files")
    
    args = parser.parse_args()
    
    # Directory containing test files
    tests_dir = args.output_dir
    
    if args.find_hyphenated:
        # Find all hyphenated model test files
        hyphenated_models = []
        
        for file in os.listdir(tests_dir):
            if file.startswith("test_hf_") and file.endswith(".py"):
                model_type = file.replace("test_hf_", "").replace(".py", "")
                if "-" in model_type:
                    hyphenated_models.append(model_type)
        
        if not hyphenated_models:
            logger.info("No hyphenated model test files found")
            return 0
        
        logger.info(f"Found {len(hyphenated_models)} hyphenated model test files: {', '.join(hyphenated_models)}")
        
        # Regenerate each hyphenated model test file
        success_count = 0
        failure_count = 0
        
        for model_type in hyphenated_models:
            logger.info(f"Regenerating test file for {model_type}...")
            success, _ = regenerate_test_file(model_type, tests_dir)
            if success:
                success_count += 1
            else:
                failure_count += 1
        
        logger.info(f"Successfully regenerated {success_count} test files")
        if failure_count > 0:
            logger.error(f"Failed to regenerate {failure_count} test files")
            return 1
    
    elif args.models:
        # Regenerate test files for specified models
        success_count = 0
        failure_count = 0
        
        for model_type in args.models:
            logger.info(f"Regenerating test file for {model_type}...")
            success, _ = regenerate_test_file(model_type, tests_dir)
            if success:
                success_count += 1
            else:
                failure_count += 1
        
        logger.info(f"Successfully regenerated {success_count} test files")
        if failure_count > 0:
            logger.error(f"Failed to regenerate {failure_count} test files")
            return 1
    
    else:
        logger.error("Please specify either --find-hyphenated or --models")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())