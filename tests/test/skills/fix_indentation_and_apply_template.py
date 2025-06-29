#!/usr/bin/env python3
"""
Fix indentation and apply template for a model type.

This script is designed to create a correctly formatted test file for any HuggingFace model type
by applying an architecture-specific template and fixing any indentation issues.

Usage:
    python fix_indentation_and_apply_template.py --model MODEL_TYPE
"""

import os
import sys
import argparse
import logging
import re
from datetime import datetime
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

# Define architecture types for model mapping
ARCHITECTURE_TYPES = {
    "encoder-only": ["bert", "distilbert", "roberta", "electra", "albert", "camembert", "xlm-roberta", "deberta"],
    "decoder-only": ["gpt2", "gpt-j", "gptj", "gpt-neo", "gpt_neo", "gpt_neox", "bloom", "llama", "mistral", "falcon", "phi", "mixtral", "mpt", "opt"],
    "encoder-decoder": ["t5", "bart", "pegasus", "mbart", "mt5", "longt5", "led", "marian"],
    "vision": ["vit", "swin", "deit", "beit", "convnext", "poolformer", "dinov2", "resnet", "segformer", "yolos", "detr", "sam", "mask2former"],
    "vision-text": ["clip", "blip"],
    "speech": ["wav2vec2", "hubert", "whisper", "bark", "speecht5", "musicgen", "encodec", "sew"],
    "multimodal": ["llava", "clip", "blip", "git", "pix2struct", "paligemma", "video-llava", "flava", "idefics", "imagebind"]
}

def get_architecture_type(model_type):
    """Determine architecture type based on model type."""
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
        "vision": os.path.join(template_dir, "minimal_vision_template.py"),  # Using our minimal vision template
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
        # Encoder-only text models
        "bert": "bert-base-uncased",
        "roberta": "roberta-base",
        "distilbert": "distilbert-base-uncased",
        "electra": "google/electra-small-discriminator",
        "albert": "albert-base-v2",
        "xlm-roberta": "xlm-roberta-base",
        "ernie": "ernie-1.0-base-zh",
        "rembert": "google/rembert",
        
        # Decoder-only text models
        "gpt2": "gpt2",
        "gptj": "EleutherAI/gpt-j-6b",
        "gpt_neo": "EleutherAI/gpt-neo-125m",
        "gpt_neox": "EleutherAI/gpt-neox-20b",
        "bloom": "bigscience/bloom-560m",
        "llama": "meta-llama/Llama-2-7b",
        "opt": "facebook/opt-125m",
        "gemma": "google/gemma-2b",
        
        # Encoder-decoder text models
        "t5": "t5-small",
        "bart": "facebook/bart-base",
        "pegasus": "google/pegasus-xsum",
        "mbart": "facebook/mbart-large-cc25",
        "mt5": "google/mt5-small",
        "prophetnet": "microsoft/prophetnet-large-uncased",
        "led": "allenai/led-base-16384",
        
        # Vision models
        "vit": "google/vit-base-patch16-224",
        "swin": "microsoft/swin-tiny-patch4-window7-224",
        "deit": "facebook/deit-base-patch16-224",
        "beit": "microsoft/beit-base-patch16-224-pt22k-ft22k",
        "convnext": "facebook/convnext-tiny-224",
        "segformer": "nvidia/segformer-b0-finetuned-ade-512-512",
        "detr": "facebook/detr-resnet-50",
        "sam": "facebook/sam-vit-base",
        "dinov2": "facebook/dinov2-base",
        "mask2former": "facebook/mask2former-swin-base-coco-instance",
        "yolos": "hustvl/yolos-small",
        
        # Multimodal models
        "clip": "openai/clip-vit-base-patch32",
        "blip": "Salesforce/blip-vqa-base",
        "llava": "llava-hf/llava-1.5-7b-hf",
        "flava": "facebook/flava-full",
        "git": "microsoft/git-base",
        "idefics": "HuggingFaceM4/idefics-9b-instruct",
        "paligemma": "google/paligemma-3b-mix-224px",
        "imagebind": "facebook/imagebind-huge",
        
        # Audio models
        "wav2vec2": "facebook/wav2vec2-base-960h",
        "hubert": "facebook/hubert-base-ls960",
        "whisper": "openai/whisper-small",
        "sew": "asapp/sew-mid-100k-ft-ls100h",
        "musicgen": "facebook/musicgen-small",
        "encodec": "facebook/encodec_24khz",
        "clap": "laion/clap-htsat-unfused",
        "unispeech": "microsoft/unispeech-large-1500h-cv"
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

def create_test_file(model_type, output_dir="fixed_tests"):
    """
    Create a test file for a specific model type.
    
    Args:
        model_type: Type of model (e.g., bert, gpt2, t5)
        output_dir: Directory to save the test file
        
    Returns:
        Tuple of (success, output_path)
    """
    try:
        # Fix for hyphenated model names
        model_valid = to_valid_identifier(model_type)
        
        # Determine output path - always use valid Python identifier
        output_path = os.path.join(output_dir, f"test_hf_{model_valid}.py")
        
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
        model_upper = model_valid.upper()
        
        # Properly capitalize model name
        if '-' in model_type:
            # Handle special capitalization for hyphenated models
            parts = model_type.split('-')
            if model_type.lower() in ["gpt-j", "gpt-neo", "gpt-neox"]:
                # GPT-J should be GPTJ, GPT-Neo should be GPTNeo
                model_capitalized = ''.join(part.upper() if i == 0 else part.capitalize() 
                                         for i, part in enumerate(parts))
            elif model_type.lower() in ["xlm-roberta"]:
                # XLM-RoBERTa should be XLMRoBERTa
                model_capitalized = 'XLMRoBERTa'
            else:
                model_capitalized = ''.join(part.capitalize() for part in parts)
        else:
            model_capitalized = model_type.capitalize()
        
        # Replace all occurrences of the model type in the content with the valid identifier
        content = template_content
        
        # Replace based on architecture type
        if arch_type == "encoder-only":
            replacements = {
                "BERT_MODELS_REGISTRY": f"{model_upper}_MODELS_REGISTRY",
                "TestBertModels": f"Test{model_capitalized}Models",
                "bert": model_valid,  # Use valid identifier
                "BertForMaskedLM": f"{model_capitalized}ForMaskedLM",
                "bert-base-uncased": default_model,
                "fill-mask": "fill-mask",
                "hf_bert_": f"hf_{model_valid}_"
            }
        elif arch_type == "decoder-only":
            replacements = {
                "GPT2_MODELS_REGISTRY": f"{model_upper}_MODELS_REGISTRY",
                "TestGpt2Models": f"Test{model_capitalized}Models",
                "gpt2": model_valid,  # Use valid identifier
                "GPT2LMHeadModel": f"{model_capitalized}LMHeadModel",
                "gpt2-medium": default_model,
                "text-generation": "text-generation",
                "hf_gpt2_": f"hf_{model_valid}_"
            }
        elif arch_type == "encoder-decoder":
            replacements = {
                "T5_MODELS_REGISTRY": f"{model_upper}_MODELS_REGISTRY",
                "TestT5Models": f"Test{model_capitalized}Models",
                "t5": model_valid,  # Use valid identifier
                "T5ForConditionalGeneration": f"{model_capitalized}ForConditionalGeneration",
                "t5-small": default_model,
                "text2text-generation": "text2text-generation",
                "hf_t5_": f"hf_{model_valid}_"
            }
        elif arch_type == "vision":
            replacements = {
                "VIT_MODELS_REGISTRY": f"{model_upper}_MODELS_REGISTRY",
                "TestVitModels": f"Test{model_capitalized}Models",
                "vit": model_valid,  # Use valid identifier
                "ViTForImageClassification": f"{model_capitalized}ForImageClassification",
                "google/vit-base-patch16-224": default_model,
                "image-classification": "image-classification",
                "hf_vit_": f"hf_{model_valid}_"
            }
        elif arch_type in ["vision-text", "multimodal"]:
            replacements = {
                "CLIP_MODELS_REGISTRY": f"{model_upper}_MODELS_REGISTRY",
                "TestClipModels": f"Test{model_capitalized}Models",
                "clip": model_valid,  # Use valid identifier
                "CLIPModel": f"{model_capitalized}Model",
                "openai/clip-vit-base-patch32": default_model,
                "image-to-text": "image-to-text",
                "hf_clip_": f"hf_{model_valid}_"
            }
        elif arch_type == "speech":
            replacements = {
                "WHISPER_MODELS_REGISTRY": f"{model_upper}_MODELS_REGISTRY",
                "TestWhisperModels": f"Test{model_capitalized}Models",
                "whisper": model_valid,  # Use valid identifier
                "WhisperForConditionalGeneration": f"{model_capitalized}ForConditionalGeneration",
                "openai/whisper-small": default_model,
                "automatic-speech-recognition": "automatic-speech-recognition",
                "hf_whisper_": f"hf_{model_valid}_"
            }
        else:
            replacements = {
                "BERT_MODELS_REGISTRY": f"{model_upper}_MODELS_REGISTRY",
                "TestBertModels": f"Test{model_capitalized}Models",
                "bert": model_valid,  # Use valid identifier
                "bert-base-uncased": default_model,
                "hf_bert_": f"hf_{model_valid}_"
            }
        
        # Apply basic replacements
        for old, new in replacements.items():
            content = content.replace(old, new)
        
        # Handle special case of model entries in registry that include hyphens
        if '-' in model_type:
            # Fix class registry entries with hyphens
            content = re.sub(
                f'"{model_type}(-[^"]*)?"\s*:\s*{{\s*"description"',
                f'"{model_type}\\1": {{"description"',
                content
            )
            
            # Fix class name references with hyphens
            content = re.sub(
                f"{model_capitalized}ForCausalLM",
                f"{model_capitalized}ForCausalLM".replace('-', ''),
                content
            )
            
            # Fix class declarations with hyphens
            content = re.sub(
                f"class Test{model_capitalized}Models",
                f"class Test{model_capitalized.replace('-', '')}Models",
                content
            )
            
            # Fix import references with hyphens
            content = re.sub(
                f"import {model_capitalized}",
                f"import {model_capitalized.replace('-', '')}",
                content
            )
            
            # Fix transformer references with hyphens
            content = re.sub(
                f"transformers.{model_capitalized}",
                f"transformers.{model_capitalized.replace('-', '')}",
                content
            )
            
            # Fix additional references in registry class values with hyphens
            content = re.sub(
                f'"class":\s*"{model_capitalized}[^"]*"',
                lambda m: m.group(0).replace('-', ''),
                content
            )
            
            # Fix tester instantiation with hyphens
            content = re.sub(
                f"tester = Test{model_capitalized}Models",
                f"tester = Test{model_capitalized.replace('-', '')}Models",
                content
            )
            
            # Keep model IDs with hyphens in most places, but convert to
            # valid identifier for variables
            content = re.sub(
                f"self.model_id = model_id or \"{model_type}\"",
                f"self.model_id = model_id or \"{model_type}\"",
                content
            )
            
            # Fix other problematic replacements
            content = content.replace(f"hf_{model_type}_", f"hf_{model_valid}_")
            content = content.replace(f"hf-{model_type}-", f"hf_{model_valid}_")
        
        # Create directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Write the file
        with open(output_path, 'w') as f:
            f.write(content)
        
        logger.info(f"Created test file: {output_path}")
        
        # Verify syntax
        try:
            compile(content, output_path, 'exec')
            logger.info(f"✅ {output_path}: Syntax is valid")
            return True, output_path
        except SyntaxError as e:
            logger.error(f"❌ {output_path}: Syntax error: {e}")
            return False, output_path
            
    except Exception as e:
        logger.error(f"Error creating test file for {model_type}: {e}")
        return False, None

def main():
    parser = argparse.ArgumentParser(description="Fix indentation and apply template for a model type")
    parser.add_argument("--model", type=str, required=True, help="Model type to generate (e.g., bert, gpt2, t5)")
    parser.add_argument("--output-dir", type=str, default="fixed_tests", help="Directory to save fixed file")
    
    args = parser.parse_args()
    
    # Create the file
    success, output_path = create_test_file(args.model, args.output_dir)
    
    if success:
        print(f"\nSuccessfully created test file for {args.model} at {output_path}")
    else:
        print(f"\nFailed to create test file for {args.model}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
