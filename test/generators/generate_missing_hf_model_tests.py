#!/usr/bin/env python3
"""
Generate Missing HuggingFace Model Tests

This script addresses Priority #2 from CLAUDE.md: "Comprehensive HuggingFace Model Testing (300+ classes)"
by generating test files for missing model families. It uses architecture-specific templates
to ensure consistent test implementation across all model types.

Features:
1. Identifies which HuggingFace model families are missing test files
2. Generates test files using appropriate templates based on architecture type
3. Properly handles hyphenated model names
4. Verifies syntax of generated files
5. Updates README in fixed_tests directory with information about new models
6. Supports filtering by architecture type and priority

Usage:
  python generate_missing_hf_model_tests.py --list  # List missing models
  python generate_missing_hf_model_tests.py --generate --priority critical  # Generate critical models
  python generate_missing_hf_model_tests.py --generate --arch encoder-only  # Generate encoder-only models
  python generate_missing_hf_model_tests.py --verify  # Verify syntax of all test files
"""

import os
import sys
import json
import re
import logging
import argparse
import importlib
import traceback
import subprocess
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"generate_tests_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    ]
)
logger = logging.getLogger(__name__)

# Path constants
ROOT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
SKILLS_DIR = ROOT_DIR / "skills"
FIXED_TESTS_DIR = SKILLS_DIR / "fixed_tests"
TEMPLATES_DIR = SKILLS_DIR / "templates"
REPORTS_DIR = ROOT_DIR / "reports"

# Create reports directory if it doesn't exist
REPORTS_DIR.mkdir(exist_ok=True)

# All HuggingFace model families with their architecture types and priorities
HUGGINGFACE_MODEL_FAMILIES = {
    # Encoder-only models
    "bert": {"architecture_type": "encoder-only", "description": "BERT masked language models", "priority": "critical"},
    "roberta": {"architecture_type": "encoder-only", "description": "RoBERTa masked language models", "priority": "critical"},
    "distilbert": {"architecture_type": "encoder-only", "description": "DistilBERT masked language models", "priority": "critical"},
    "albert": {"architecture_type": "encoder-only", "description": "ALBERT masked language models", "priority": "high"},
    "electra": {"architecture_type": "encoder-only", "description": "ELECTRA discriminator models", "priority": "high"},
    "camembert": {"architecture_type": "encoder-only", "description": "CamemBERT masked language models", "priority": "medium"},
    "xlm-roberta": {"architecture_type": "encoder-only", "description": "XLM-RoBERTa masked language models", "priority": "high"},
    "deberta": {"architecture_type": "encoder-only", "description": "DeBERTa masked language models", "priority": "high"},
    "deberta-v2": {"architecture_type": "encoder-only", "description": "DeBERTa-v2 masked language models", "priority": "high"},
    "xlnet": {"architecture_type": "encoder-only", "description": "XLNet autoregressive language models", "priority": "medium"},
    "reformer": {"architecture_type": "encoder-only", "description": "Reformer efficient transformer models", "priority": "medium"},
    "layoutlm": {"architecture_type": "encoder-only", "description": "LayoutLM document understanding models", "priority": "medium"},
    "layoutlmv2": {"architecture_type": "encoder-only", "description": "LayoutLMv2 document understanding models", "priority": "medium"},
    "layoutlmv3": {"architecture_type": "encoder-only", "description": "LayoutLMv3 document understanding models", "priority": "medium"},
    "roformer": {"architecture_type": "encoder-only", "description": "RoFormer rotary position embedding models", "priority": "medium"},
    
    # Decoder-only models
    "gpt2": {"architecture_type": "decoder-only", "description": "GPT-2 autoregressive language models", "priority": "critical"},
    "gpt-j": {"architecture_type": "decoder-only", "description": "GPT-J autoregressive language models", "priority": "high"},
    "gpt-neo": {"architecture_type": "decoder-only", "description": "GPT-Neo autoregressive language models", "priority": "high"},
    "gpt-neox": {"architecture_type": "decoder-only", "description": "GPT-NeoX autoregressive language models", "priority": "high"},
    "bloom": {"architecture_type": "decoder-only", "description": "BLOOM multilingual autoregressive language models", "priority": "high"},
    "llama": {"architecture_type": "decoder-only", "description": "LLaMA autoregressive language models", "priority": "critical"},
    "mistral": {"architecture_type": "decoder-only", "description": "Mistral autoregressive language models", "priority": "critical"},
    "falcon": {"architecture_type": "decoder-only", "description": "Falcon autoregressive language models", "priority": "high"},
    "phi": {"architecture_type": "decoder-only", "description": "Phi small language models", "priority": "high"},
    "mixtral": {"architecture_type": "decoder-only", "description": "Mixtral mixture of experts models", "priority": "high"},
    "mpt": {"architecture_type": "decoder-only", "description": "MPT autoregressive language models", "priority": "medium"},
    "ctrl": {"architecture_type": "decoder-only", "description": "CTRL controllable language models", "priority": "medium"},
    "opt": {"architecture_type": "decoder-only", "description": "OPT Open Pre-trained Transformers", "priority": "high"},
    "gemma": {"architecture_type": "decoder-only", "description": "Gemma lightweight models", "priority": "high"},
    "codegen": {"architecture_type": "decoder-only", "description": "CodeGen code generation models", "priority": "medium"},
    "stablelm": {"architecture_type": "decoder-only", "description": "StableLM autoregressive language models", "priority": "medium"},
    
    # Encoder-decoder models
    "t5": {"architecture_type": "encoder-decoder", "description": "T5 text-to-text transfer transformer models", "priority": "critical"},
    "bart": {"architecture_type": "encoder-decoder", "description": "BART sequence-to-sequence models", "priority": "critical"},
    "pegasus": {"architecture_type": "encoder-decoder", "description": "Pegasus summarization models", "priority": "high"},
    "mbart": {"architecture_type": "encoder-decoder", "description": "Multilingual BART models", "priority": "high"},
    "longt5": {"architecture_type": "encoder-decoder", "description": "LongT5 long sequence text-to-text models", "priority": "medium"},
    "led": {"architecture_type": "encoder-decoder", "description": "Longformer Encoder-Decoder models", "priority": "medium"},
    "marian": {"architecture_type": "encoder-decoder", "description": "Marian machine translation models", "priority": "medium"},
    "mt5": {"architecture_type": "encoder-decoder", "description": "Multilingual T5 models", "priority": "high"},
    "flan-t5": {"architecture_type": "encoder-decoder", "description": "Flan-T5 instruction-tuned models", "priority": "high"},
    "flan-ul2": {"architecture_type": "encoder-decoder", "description": "Flan-UL2 unified language learner", "priority": "medium"},
    "prophetnet": {"architecture_type": "encoder-decoder", "description": "ProphetNet sequence-to-sequence models", "priority": "medium"},
    "bigbird": {"architecture_type": "encoder-decoder", "description": "BigBird long sequence models", "priority": "medium"},
    
    # Vision models
    "vit": {"architecture_type": "vision", "description": "Vision Transformer models", "priority": "critical"},
    "swin": {"architecture_type": "vision", "description": "Swin Transformer vision models", "priority": "high"},
    "deit": {"architecture_type": "vision", "description": "Data-efficient Image Transformers", "priority": "high"},
    "beit": {"architecture_type": "vision", "description": "BERT pre-training for Image Transformers", "priority": "high"},
    "convnext": {"architecture_type": "vision", "description": "ConvNeXT modern CNN models", "priority": "high"},
    "poolformer": {"architecture_type": "vision", "description": "PoolFormer MetaFormer models", "priority": "medium"},
    "dinov2": {"architecture_type": "vision", "description": "DINOv2 self-supervised vision models", "priority": "high"},
    "detr": {"architecture_type": "vision", "description": "Detection Transformer object detection", "priority": "high"},
    "resnet": {"architecture_type": "vision", "description": "ResNet convolutional neural networks", "priority": "medium"},
    "segformer": {"architecture_type": "vision", "description": "SegFormer image segmentation models", "priority": "medium"},
    "maskformer": {"architecture_type": "vision", "description": "MaskFormer segmentation models", "priority": "medium"},
    "sam": {"architecture_type": "vision", "description": "Segment Anything Model", "priority": "high"},
    "yolos": {"architecture_type": "vision", "description": "YOLOS object detection transformers", "priority": "medium"},
    
    # Speech models
    "wav2vec2": {"architecture_type": "speech", "description": "Wav2Vec2 speech recognition models", "priority": "high"},
    "hubert": {"architecture_type": "speech", "description": "HuBERT speech models", "priority": "high"},
    "whisper": {"architecture_type": "speech", "description": "Whisper speech recognition models", "priority": "critical"},
    "speecht5": {"architecture_type": "speech", "description": "SpeechT5 unified speech models", "priority": "medium"},
    "unispeech": {"architecture_type": "speech", "description": "UniSpeech speech models", "priority": "medium"},
    "unispeech-sat": {"architecture_type": "speech", "description": "UniSpeech-SAT speech models", "priority": "medium"},
    "encodec": {"architecture_type": "speech", "description": "EnCodec neural audio codec", "priority": "medium"},
    "musicgen": {"architecture_type": "speech", "description": "MusicGen music generation models", "priority": "medium"},
    
    # Multimodal models
    "clip": {"architecture_type": "multimodal", "description": "CLIP vision-language models", "priority": "critical"},
    "blip": {"architecture_type": "multimodal", "description": "BLIP vision-language models", "priority": "high"},
    "blip-2": {"architecture_type": "multimodal", "description": "BLIP-2 vision-language models", "priority": "high"},
    "llava": {"architecture_type": "multimodal", "description": "LLaVA multimodal language-vision models", "priority": "high"},
    "git": {"architecture_type": "multimodal", "description": "GIT Generative Image-to-text models", "priority": "medium"},
    "pix2struct": {"architecture_type": "multimodal", "description": "Pix2Struct image-to-text models", "priority": "medium"},
    "paligemma": {"architecture_type": "multimodal", "description": "Paligemma multimodal models", "priority": "medium"},
    "video-llava": {"architecture_type": "multimodal", "description": "Video-LLaVA video-language models", "priority": "medium"},
    "donut": {"architecture_type": "multimodal", "description": "DONUT document understanding models", "priority": "medium"},
    "bridgetower": {"architecture_type": "multimodal", "description": "BridgeTower vision-language models", "priority": "medium"},
    "flava": {"architecture_type": "multimodal", "description": "FLAVA multimodal foundation models", "priority": "medium"},
    "imagebind": {"architecture_type": "multimodal", "description": "ImageBind multimodal models", "priority": "medium"},
}

# Class name fixes for proper capitalization
CLASS_NAME_FIXES = {
    "VitForImageClassification": "ViTForImageClassification",
    "SwinForImageClassification": "SwinForImageClassification",
    "DeitForImageClassification": "DeiTForImageClassification",
    "BeitForImageClassification": "BEiTForImageClassification",
    "ConvnextForImageClassification": "ConvNextForImageClassification",
    "Gpt2LMHeadModel": "GPT2LMHeadModel",
    "GptjForCausalLM": "GPTJForCausalLM",
    "GptneoForCausalLM": "GPTNeoForCausalLM",
    "GptneoxForCausalLM": "GPTNeoXForCausalLM",
    "XlmRobertaForMaskedLM": "XLMRobertaForMaskedLM",
    "XlmRobertaModel": "XLMRobertaModel",
    "RobertaForMaskedLM": "RobertaForMaskedLM",
    "DistilbertForMaskedLM": "DistilBertForMaskedLM",
    "AlbertForMaskedLM": "AlbertForMaskedLM",
    "ElectraForMaskedLM": "ElectraForMaskedLM",
    "BartForConditionalGeneration": "BartForConditionalGeneration",
    "MbartForConditionalGeneration": "MBartForConditionalGeneration",
    "PegasusForConditionalGeneration": "PegasusForConditionalGeneration",
    "Mt5ForConditionalGeneration": "MT5ForConditionalGeneration",
    "ClipModel": "CLIPModel",
    "BlipForConditionalGeneration": "BlipForConditionalGeneration",
    "Blip2ForConditionalGeneration": "Blip2ForConditionalGeneration",
    "LlavaForConditionalGeneration": "LlavaForConditionalGeneration",
    "WhisperForConditionalGeneration": "WhisperForConditionalGeneration",
    "Wav2vec2ForCTC": "Wav2Vec2ForCTC",
    "HubertForCTC": "HubertForCTC",
    "LlamaForCausalLM": "LlamaForCausalLM",
    "OptForCausalLM": "OPTForCausalLM",
    "BloomForCausalLM": "BloomForCausalLM",
    "FalconForCausalLM": "FalconForCausalLM",
    "MistralForCausalLM": "MistralForCausalLM",
    "GemmaForCausalLM": "GemmaForCausalLM",
    "CodegenForCausalLM": "CodeGenForCausalLM",
    "MptForCausalLM": "MPTForCausalLM",
}

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

def get_class_name(model_family):
    """Get the properly capitalized class name for a model family."""
    if "-" in model_family:
        # Handle hyphenated model names by capitalizing each part
        model_capitalized = ''.join(part.capitalize() for part in model_family.split('-'))
    else:
        model_capitalized = model_family.capitalize()
    
    # Return the class name based on architecture type
    architecture_type = HUGGINGFACE_MODEL_FAMILIES.get(model_family, {}).get("architecture_type", "encoder-only")
    
    if architecture_type == "encoder-only":
        class_name = f"{model_capitalized}ForMaskedLM"
    elif architecture_type == "decoder-only":
        class_name = f"{model_capitalized}ForCausalLM"
    elif architecture_type == "encoder-decoder":
        class_name = f"{model_capitalized}ForConditionalGeneration"
    elif architecture_type == "vision":
        class_name = f"{model_capitalized}ForImageClassification"
    elif architecture_type == "speech":
        class_name = f"{model_capitalized}ForCTC"
        # Special case for Whisper
        if model_family.lower() == "whisper":
            class_name = "WhisperForConditionalGeneration"
    elif architecture_type == "multimodal":
        if model_family.lower() == "clip":
            class_name = "CLIPModel"
        elif "llava" in model_family.lower():
            class_name = "LlavaForConditionalGeneration"
        elif "blip" in model_family.lower():
            class_name = "BlipForConditionalGeneration"
        else:
            class_name = f"{model_capitalized}Model"
    else:
        class_name = f"{model_capitalized}Model"
    
    # Apply class name fixes if available
    for old, new in CLASS_NAME_FIXES.items():
        if old.lower() == class_name.lower():
            return new
    
    return class_name

def get_existing_test_files():
    """Get a list of existing test files in the fixed_tests directory."""
    test_files = []
    if FIXED_TESTS_DIR.exists():
        test_files = list(FIXED_TESTS_DIR.glob("test_hf_*.py"))
    return [f.stem for f in test_files]

def extract_model_family_from_filename(filename):
    """Extract model family from a test file name."""
    if filename.startswith("test_hf_"):
        # Remove test_hf_ prefix
        family = filename[8:]
        # Convert back to hyphenated form if needed
        family = family.replace("_", "-")
        return family
    return None

def get_model_families_with_tests():
    """Get a set of model families that already have test files."""
    test_files = get_existing_test_files()
    families = set()
    
    for filename in test_files:
        family = extract_model_family_from_filename(filename)
        if family:
            # Convert underscores back to hyphens for comparison
            family = family.replace("_", "-")
            families.add(family)
    
    return families

def identify_missing_model_families():
    """Identify model families that don't have test files yet."""
    families_with_tests = get_model_families_with_tests()
    all_families = set(HUGGINGFACE_MODEL_FAMILIES.keys())
    
    missing_families = all_families - families_with_tests
    return sorted(missing_families)

def generate_test_file(model_family, output_dir=None):
    """Generate a test file for a model family using architecture-specific templates."""
    if output_dir is None:
        output_dir = FIXED_TESTS_DIR
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get architecture type and priority
    model_info = HUGGINGFACE_MODEL_FAMILIES.get(model_family, {})
    arch_type = model_info.get("architecture_type", "encoder-only")
    
    # Find appropriate template
    template_map = {
        "encoder-only": TEMPLATES_DIR / "encoder_only_template.py",
        "decoder-only": TEMPLATES_DIR / "decoder_only_template.py",
        "encoder-decoder": TEMPLATES_DIR / "encoder_decoder_template.py",
        "vision": TEMPLATES_DIR / "vision_template.py",
        "speech": TEMPLATES_DIR / "speech_template.py",
        "multimodal": TEMPLATES_DIR / "multimodal_template.py"
    }
    
    template_path = template_map.get(arch_type)
    if not template_path or not template_path.exists():
        logger.error(f"Template not found for {arch_type}")
        # Fallback to encoder-only template
        template_path = TEMPLATES_DIR / "encoder_only_template.py"
        if not template_path.exists():
            logger.error(f"Fallback template not found")
            return False
    
    try:
        # Read template
        with open(template_path, "r") as f:
            template_content = f.read()
        
        # Prepare model metadata
        model_family_valid = to_valid_identifier(model_family)
        model_family_upper = model_family_valid.upper()
        
        if "-" in model_family:
            # Handle hyphenated model names by capitalizing each part
            model_capitalized = ''.join(part.capitalize() for part in model_family.split('-'))
        else:
            model_capitalized = model_family.capitalize()
        
        test_class = f"Test{model_capitalized}Models"
        module_name = f"test_hf_{model_family_valid}"
        
        # Get properly capitalized class name
        class_name = get_class_name(model_family)
        
        # Determine default model ID
        default_model = f"{model_family}"
        # Add organization prefix if not present
        if "/" not in default_model:
            if "gpt-j" in model_family:
                default_model = f"EleutherAI/{model_family}"
            elif "gpt-neo" in model_family:
                default_model = f"EleutherAI/{model_family}"
            elif "clip" in model_family:
                default_model = f"openai/{model_family}"
            elif "whisper" in model_family:
                default_model = f"openai/{model_family}-small"
            elif "vit" in model_family:
                default_model = f"google/{model_family}-base-patch16-224"
        
        # Determine default task
        task_map = {
            "encoder-only": "fill-mask",
            "decoder-only": "text-generation",
            "encoder-decoder": "text2text-generation",
            "vision": "image-classification",
            "speech": "automatic-speech-recognition",
            "multimodal": "zero-shot-image-classification"
        }
        default_task = task_map.get(arch_type, "fill-mask")
        
        # Example input text based on architecture
        input_text_map = {
            "encoder-only": f"{model_capitalized} is a <mask> language model.",
            "decoder-only": f"{model_capitalized} is a transformer model that",
            "encoder-decoder": f"translate English to German: The house is wonderful.",
            "vision": "", # No text input for vision models
            "speech": "", # No text input for speech models
            "multimodal": "" # Complex inputs for multimodal models
        }
        default_input_text = input_text_map.get(arch_type, f"{model_capitalized} is a model that")
        
        # Make all necessary replacements based on model type and architecture
        # First determine which template we're working with by checking content
        replacements = {}
        
        # Look for specific registry names to determine template type
        if "BERT_MODELS_REGISTRY" in template_content:
            # Encoder-only template (BERT-based)
            replacements = {
                "BERT_MODELS_REGISTRY": f"{model_family_upper}_MODELS_REGISTRY",
                "TestBertModels": test_class,
                "bert-base-uncased": default_model,
                "BERT": model_family_upper,
                "BertForMaskedLM": class_name,
                "bert": model_family_valid,
                "fill-mask": default_task,
                "The quick brown fox jumps over the [MASK] dog.": default_input_text,
                "hf_bert_": f"hf_{model_family_valid}_"
            }
        elif "GPT2_MODELS_REGISTRY" in template_content:
            # Decoder-only template (GPT2-based)
            replacements = {
                "GPT2_MODELS_REGISTRY": f"{model_family_upper}_MODELS_REGISTRY",
                "TestGpt2Models": test_class,
                "gpt2": default_model,
                "GPT2": model_family_upper,
                "GPT2LMHeadModel": class_name,
                "gpt2": model_family_valid,
                "text-generation": default_task,
                "GPT-2 is a transformer model that": default_input_text,
                "hf_gpt2_": f"hf_{model_family_valid}_"
            }
        elif "T5_MODELS_REGISTRY" in template_content:
            # Encoder-decoder template (T5-based)
            replacements = {
                "T5_MODELS_REGISTRY": f"{model_family_upper}_MODELS_REGISTRY",
                "TestT5Models": test_class,
                "t5-small": default_model,
                "T5": model_family_upper,
                "T5ForConditionalGeneration": class_name,
                "t5": model_family_valid,
                "text2text-generation": default_task,
                "translate English to German: The house is wonderful.": default_input_text,
                "hf_t5_": f"hf_{model_family_valid}_"
            }
        elif "VIT_MODELS_REGISTRY" in template_content:
            # Vision template (ViT-based)
            replacements = {
                "VIT_MODELS_REGISTRY": f"{model_family_upper}_MODELS_REGISTRY",
                "TestVitModels": test_class,
                "google/vit-base-patch16-224": default_model,
                "VIT": model_family_upper,
                "ViTForImageClassification": class_name,
                "vit": model_family_valid,
                "image-classification": default_task,
                "hf_vit_": f"hf_{model_family_valid}_"
            }
        elif "WHISPER_MODELS_REGISTRY" in template_content:
            # Speech template (Whisper-based)
            replacements = {
                "WHISPER_MODELS_REGISTRY": f"{model_family_upper}_MODELS_REGISTRY",
                "TestWhisperModels": test_class,
                "openai/whisper-small": default_model,
                "WHISPER": model_family_upper,
                "WhisperForConditionalGeneration": class_name,
                "whisper": model_family_valid,
                "automatic-speech-recognition": default_task,
                "hf_whisper_": f"hf_{model_family_valid}_"
            }
        elif "CLIP_MODELS_REGISTRY" in template_content:
            # Multimodal template (CLIP-based)
            replacements = {
                "CLIP_MODELS_REGISTRY": f"{model_family_upper}_MODELS_REGISTRY",
                "TestClipModels": test_class,
                "openai/clip-vit-base-patch32": default_model,
                "CLIP": model_family_upper,
                "CLIPModel": class_name,
                "clip": model_family_valid,
                "zero-shot-image-classification": default_task,
                "hf_clip_": f"hf_{model_family_valid}_"
            }
        
        # Create the test content with replacements
        content = template_content
        for old, new in replacements.items():
            content = content.replace(old, new)
        
        # Write the test file
        output_file = output_dir / f"{module_name}.py"
        with open(output_file, "w") as f:
            f.write(content)
        
        # Validate syntax
        try:
            compile(content, str(output_file), "exec")
            logger.info(f"✅ Syntax is valid for {output_file}")
            return True
        except SyntaxError as e:
            logger.error(f"❌ Syntax error in generated file: {e}")
            logger.error(f"Line {e.lineno}: {e.text}")
            return False
            
    except Exception as e:
        logger.error(f"Error generating test file for {model_family}: {e}")
        traceback.print_exc()
        return False

def generate_missing_tests(arch_types=None, priorities=None, max_models=None):
    """Generate test files for missing model families."""
    missing_families = identify_missing_model_families()
    
    # Filter by architecture type if specified
    if arch_types:
        missing_families = [f for f in missing_families 
                          if HUGGINGFACE_MODEL_FAMILIES.get(f, {}).get("architecture_type") in arch_types]
    
    # Filter by priority if specified
    if priorities:
        missing_families = [f for f in missing_families 
                          if HUGGINGFACE_MODEL_FAMILIES.get(f, {}).get("priority") in priorities]
    
    # Limit to max_models if specified
    if max_models is not None and max_models > 0:
        missing_families = missing_families[:max_models]
    
    logger.info(f"Generating test files for {len(missing_families)} missing model families")
    
    # Track successful generations
    successful_generations = []
    failed_generations = []
    
    for i, family in enumerate(missing_families):
        logger.info(f"[{i+1}/{len(missing_families)}] Generating test for {family}")
        success = generate_test_file(family)
        
        if success:
            logger.info(f"✅ Successfully generated test for {family}")
            successful_generations.append(family)
        else:
            logger.error(f"❌ Failed to generate test for {family}")
            failed_generations.append(family)
    
    # Update README if any tests were generated
    if successful_generations:
        update_readme(successful_generations)
    
    # Print summary
    logger.info(f"\nGeneration Summary:")
    logger.info(f"- Total model families: {len(missing_families)}")
    logger.info(f"- Successfully generated: {len(successful_generations)}")
    logger.info(f"- Failed to generate: {len(failed_generations)}")
    
    if failed_generations:
        logger.info(f"\nFailed generations:")
        for family in failed_generations:
            logger.info(f"- {family}")
    
    return {
        "total": len(missing_families),
        "successful": successful_generations,
        "failed": failed_generations
    }

def update_readme(new_models):
    """Update the README.md in the fixed_tests directory with information about new models."""
    readme_path = FIXED_TESTS_DIR / "README.md"
    
    # Default content if README doesn't exist
    default_content = """# Fixed Tests for HuggingFace Models

This directory contains test files for HuggingFace models that handle various architectures properly.

## Model Coverage

The tests in this directory cover the following model architectures:

1. **Encoder-only models**: BERT, RoBERTa, DistilBERT, etc.
2. **Decoder-only models**: GPT-2, LLaMA, etc.
3. **Encoder-decoder models**: T5, BART, etc.
4. **Vision models**: ViT, DETR, etc.
5. **Speech models**: Whisper, Wav2Vec2, etc.
6. **Multimodal models**: CLIP, BLIP, etc.

## Running Tests

Tests can be run individually with:

```bash
python skills/fixed_tests/test_hf_bert.py --model bert-base-uncased --device cpu
```

Or you can run all tests with:

```bash
python run_comprehensive_hf_model_test.py --all
```

## Coverage Status

| Architecture Type | Models Covered |
|------------------|----------------|
"""
    
    # Read existing README if it exists
    if readme_path.exists():
        with open(readme_path, "r") as f:
            content = f.read()
    else:
        content = default_content
    
    # Get all test files
    model_families_with_tests = get_model_families_with_tests()
    
    # Group by architecture type
    arch_models = defaultdict(list)
    for family in model_families_with_tests:
        arch_type = HUGGINGFACE_MODEL_FAMILIES.get(family, {}).get("architecture_type", "unknown")
        arch_models[arch_type].append(family)
    
    # Update or create the coverage table
    table_header = "| Architecture Type | Models Covered |"
    table_separator = "|------------------|----------------|"
    
    # Check if table exists
    if table_header in content:
        # Replace existing table
        table_start = content.find(table_header)
        table_end = content.find("\n\n", table_start)
        if table_end == -1:
            table_end = len(content)
        
        table_content = table_header + "\n" + table_separator + "\n"
        for arch_type, models in sorted(arch_models.items()):
            models_str = ", ".join(sorted(models))
            table_content += f"| {arch_type} | {models_str} |\n"
        
        content = content[:table_start] + table_content + content[table_end:]
    else:
        # Add table at the end
        content += "\n## Coverage Status\n\n"
        content += table_header + "\n" + table_separator + "\n"
        for arch_type, models in sorted(arch_models.items()):
            models_str = ", ".join(sorted(models))
            content += f"| {arch_type} | {models_str} |\n"
    
    # Add section for newly added models
    if new_models:
        # Check if section exists
        if "## Recently Added Models" in content:
            # Update existing section
            section_start = content.find("## Recently Added Models")
            section_end = content.find("\n##", section_start)
            if section_end == -1:
                section_end = len(content)
            
            section_content = f"## Recently Added Models\n\nThe following models were recently added:\n\n"
            for family in sorted(new_models):
                arch_type = HUGGINGFACE_MODEL_FAMILIES.get(family, {}).get("architecture_type", "unknown")
                description = HUGGINGFACE_MODEL_FAMILIES.get(family, {}).get("description", "")
                section_content += f"- **{family}** ({arch_type}): {description}\n"
            
            content = content[:section_start] + section_content + content[section_end:]
        else:
            # Add section at the end
            content += "\n## Recently Added Models\n\nThe following models were recently added:\n\n"
            for family in sorted(new_models):
                arch_type = HUGGINGFACE_MODEL_FAMILIES.get(family, {}).get("architecture_type", "unknown")
                description = HUGGINGFACE_MODEL_FAMILIES.get(family, {}).get("description", "")
                content += f"- **{family}** ({arch_type}): {description}\n"
    
    # Write updated README
    with open(readme_path, "w") as f:
        f.write(content)
    
    logger.info(f"Updated README at {readme_path}")
    return True

def verify_test_files():
    """Verify the syntax of all test files in the fixed_tests directory."""
    if not FIXED_TESTS_DIR.exists():
        logger.error(f"Fixed tests directory not found: {FIXED_TESTS_DIR}")
        return False
    
    # Get all test files
    test_files = list(FIXED_TESTS_DIR.glob("test_hf_*.py"))
    logger.info(f"Verifying syntax of {len(test_files)} test files")
    
    # Track results
    valid_files = []
    invalid_files = []
    
    for test_file in test_files:
        try:
            with open(test_file, "r") as f:
                content = f.read()
            
            # Verify syntax
            compile(content, str(test_file), "exec")
            logger.info(f"✅ Syntax is valid for {test_file.name}")
            valid_files.append(test_file.name)
        except SyntaxError as e:
            logger.error(f"❌ Syntax error in {test_file.name}: {e}")
            invalid_files.append((test_file.name, f"Syntax error in line {e.lineno}: {e.text}"))
        except Exception as e:
            logger.error(f"❌ Error reading {test_file.name}: {e}")
            invalid_files.append((test_file.name, str(e)))
    
    # Generate report
    logger.info(f"\nVerification Summary:")
    logger.info(f"- Total test files: {len(test_files)}")
    logger.info(f"- Valid files: {len(valid_files)}")
    logger.info(f"- Invalid files: {len(invalid_files)}")
    
    if invalid_files:
        logger.info(f"\nInvalid files:")
        for name, error in invalid_files:
            logger.info(f"- {name}: {error}")
    
    # Save report
    report_file = REPORTS_DIR / f"syntax_verification_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    
    with open(report_file, "w") as f:
        f.write("# Test File Syntax Verification Report\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Summary\n\n")
        f.write(f"- Total test files: {len(test_files)}\n")
        f.write(f"- Valid files: {len(valid_files)}\n")
        f.write(f"- Invalid files: {len(invalid_files)}\n\n")
        
        if invalid_files:
            f.write("## Invalid Files\n\n")
            for name, error in invalid_files:
                f.write(f"### {name}\n\n")
                f.write(f"Error: {error}\n\n")
        
        f.write("## Valid Files\n\n")
        for name in valid_files:
            f.write(f"- {name}\n")
    
    logger.info(f"Verification report saved to {report_file}")
    
    return len(invalid_files) == 0

def find_high_priority_missing_families():
    """Identify high-priority model families that don't have test files."""
    missing_families = identify_missing_model_families()
    
    # Group by priority and architecture type
    priority_arch_models = defaultdict(lambda: defaultdict(list))
    
    for family in missing_families:
        model_info = HUGGINGFACE_MODEL_FAMILIES.get(family, {})
        priority = model_info.get("priority", "medium")
        arch_type = model_info.get("architecture_type", "unknown")
        
        priority_arch_models[priority][arch_type].append(family)
    
    # Generate report
    report_file = REPORTS_DIR / f"missing_models_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    
    with open(report_file, "w") as f:
        f.write("# Missing HuggingFace Model Tests\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Summary\n\n")
        f.write(f"Total missing model families: {len(missing_families)}\n\n")
        
        for priority in ["critical", "high", "medium"]:
            priority_count = sum(len(models) for models in priority_arch_models[priority].values())
            f.write(f"- {priority.capitalize()} priority: {priority_count} models\n")
        
        f.write("\n## Critical Priority Missing Models\n\n")
        if not priority_arch_models["critical"]:
            f.write("No critical priority models missing.\n\n")
        else:
            for arch_type, models in sorted(priority_arch_models["critical"].items()):
                f.write(f"### {arch_type}\n\n")
                for family in sorted(models):
                    description = HUGGINGFACE_MODEL_FAMILIES.get(family, {}).get("description", "")
                    f.write(f"- **{family}**: {description}\n")
                f.write("\n")
        
        f.write("## High Priority Missing Models\n\n")
        if not priority_arch_models["high"]:
            f.write("No high priority models missing.\n\n")
        else:
            for arch_type, models in sorted(priority_arch_models["high"].items()):
                f.write(f"### {arch_type}\n\n")
                for family in sorted(models):
                    description = HUGGINGFACE_MODEL_FAMILIES.get(family, {}).get("description", "")
                    f.write(f"- **{family}**: {description}\n")
                f.write("\n")
        
        f.write("## Medium Priority Missing Models\n\n")
        if not priority_arch_models["medium"]:
            f.write("No medium priority models missing.\n\n")
        else:
            for arch_type, models in sorted(priority_arch_models["medium"].items()):
                f.write(f"### {arch_type}\n\n")
                for family in sorted(models):
                    description = HUGGINGFACE_MODEL_FAMILIES.get(family, {}).get("description", "")
                    f.write(f"- **{family}**: {description}\n")
                f.write("\n")
    
    logger.info(f"Missing models report saved to {report_file}")
    
    return {
        "priority_models": {
            "critical": list(sum(priority_arch_models["critical"].values(), [])),
            "high": list(sum(priority_arch_models["high"].values(), [])),
            "medium": list(sum(priority_arch_models["medium"].values(), []))
        },
        "report_file": str(report_file)
    }

def main():
    parser = argparse.ArgumentParser(description="Generate Missing HuggingFace Model Tests")
    
    # Action options
    action_group = parser.add_mutually_exclusive_group(required=True)
    action_group.add_argument("--list", action="store_true", 
                            help="List missing model families")
    action_group.add_argument("--generate", action="store_true", 
                            help="Generate test files for missing model families")
    action_group.add_argument("--verify", action="store_true", 
                            help="Verify syntax of all test files")
    action_group.add_argument("--update-readme", action="store_true", 
                            help="Update README in fixed_tests directory")
    action_group.add_argument("--find-high-priority", action="store_true",
                            help="Find high-priority missing model families")
    
    # Filter options
    parser.add_argument("--arch", type=str, nargs="+", 
                      choices=["encoder-only", "decoder-only", "encoder-decoder", "vision", "speech", "multimodal"],
                      help="Filter by architecture type")
    parser.add_argument("--priority", type=str, nargs="+", 
                      choices=["critical", "high", "medium"],
                      help="Filter by priority level")
    parser.add_argument("--max-models", type=int, 
                      help="Maximum number of models to generate")
    
    args = parser.parse_args()
    
    if args.list:
        # List missing model families
        missing_families = identify_missing_model_families()
        
        # Filter by architecture type if specified
        if args.arch:
            missing_families = [f for f in missing_families 
                              if HUGGINGFACE_MODEL_FAMILIES.get(f, {}).get("architecture_type") in args.arch]
        
        # Filter by priority if specified
        if args.priority:
            missing_families = [f for f in missing_families 
                              if HUGGINGFACE_MODEL_FAMILIES.get(f, {}).get("priority") in args.priority]
        
        # Group by architecture type
        arch_models = defaultdict(list)
        for family in missing_families:
            arch_type = HUGGINGFACE_MODEL_FAMILIES.get(family, {}).get("architecture_type", "unknown")
            arch_models[arch_type].append(family)
        
        # Print summary
        print(f"\nMissing Model Families: {len(missing_families)}")
        
        for arch_type, models in sorted(arch_models.items()):
            print(f"\n{arch_type.upper()} ({len(models)} models):")
            
            # Group by priority
            by_priority = defaultdict(list)
            for family in models:
                priority = HUGGINGFACE_MODEL_FAMILIES.get(family, {}).get("priority", "medium")
                by_priority[priority].append(family)
            
            # Print by priority
            for priority in ["critical", "high", "medium"]:
                if by_priority[priority]:
                    print(f"\n  {priority.upper()} PRIORITY:")
                    for family in sorted(by_priority[priority]):
                        description = HUGGINGFACE_MODEL_FAMILIES.get(family, {}).get("description", "")
                        print(f"    - {family}: {description}")
    
    elif args.generate:
        # Generate test files for missing model families
        generate_missing_tests(
            arch_types=args.arch,
            priorities=args.priority,
            max_models=args.max_models
        )
    
    elif args.verify:
        # Verify syntax of all test files
        verify_test_files()
    
    elif args.update_readme:
        # Update README in fixed_tests directory
        update_readme([])
    
    elif args.find_high_priority:
        # Find high-priority missing model families
        results = find_high_priority_missing_families()
        
        # Print summary
        critical_count = len(results["priority_models"]["critical"])
        high_count = len(results["priority_models"]["high"])
        
        print(f"\nHigh-Priority Missing Models:")
        print(f"- Critical priority: {critical_count} models")
        print(f"- High priority: {high_count} models")
        
        # Print critical models
        if critical_count > 0:
            print("\nCritical Models to Implement First:")
            for i, family in enumerate(results["priority_models"]["critical"][:5], 1):
                arch_type = HUGGINGFACE_MODEL_FAMILIES.get(family, {}).get("architecture_type", "unknown")
                print(f"  {i}. {family} ({arch_type})")
        
        print(f"\nSee full report at: {results['report_file']}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())