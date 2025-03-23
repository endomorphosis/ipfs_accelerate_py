#!/usr/bin/env python3
"""
Architecture detector for HuggingFace models.

This module provides functions to detect the architecture type of a model
based on its name or structure.
"""

import re
import os
import sys
import logging
from typing import Dict, List, Any, Optional, Union, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define supported architecture types
ARCHITECTURE_TYPES = [
    "encoder-only",
    "decoder-only",
    "encoder-decoder",
    "vision",
    "vision-encoder-text-decoder",
    "speech",
    "multimodal"
]

# Define model name to architecture type mapping
MODEL_NAME_MAPPING = {
    # Encoder-only Models
    "bert": "encoder-only",
    "roberta": "encoder-only",
    "albert": "encoder-only",
    "distilbert": "encoder-only",
    "electra": "encoder-only",
    "deberta": "encoder-only",
    "funnel": "encoder-only",
    "xlm-roberta": "encoder-only",
    "camembert": "encoder-only",
    "flaubert": "encoder-only",
    "ernie": "encoder-only",
    "layoutlm": "encoder-only",
    "layoutlmv2": "encoder-only",
    "layoutlmv3": "encoder-only",
    "markuplm": "encoder-only",
    "roformer": "encoder-only",
    "longformer": "encoder-only",
    "squeezebert": "encoder-only",
    "canine": "encoder-only",
    "mpnet": "encoder-only",
    "mobilebert": "encoder-only",
    "tapas": "encoder-only",
    "bigbird": "encoder-only",
    "convbert": "encoder-only",
    "megatron-bert": "encoder-only",
    "xlm": "encoder-only",
    "rembert": "encoder-only",
    
    # Decoder-only Models
    "gpt2": "decoder-only",
    "gpt-2": "decoder-only",
    "gpt_2": "decoder-only",
    "gpt-neo": "decoder-only",
    "gpt_neo": "decoder-only",
    "gptneo": "decoder-only",
    "gpt-j": "decoder-only",
    "gpt_j": "decoder-only",
    "gptj": "decoder-only",
    "bloom": "decoder-only",
    "opt": "decoder-only",
    "ctrl": "decoder-only",
    "transfo-xl": "decoder-only",
    "transfoxl": "decoder-only",
    "transformerxl": "decoder-only",
    "xlnet": "decoder-only",
    "reformer": "decoder-only",
    "llama": "decoder-only",
    "llama2": "decoder-only",
    "llama-2": "decoder-only",
    "llama3": "decoder-only",
    "llama-3": "decoder-only",
    "mpt": "decoder-only",
    "falcon": "decoder-only",
    "mistral": "decoder-only",
    "phi": "decoder-only",
    "gemma": "decoder-only",
    "phi-2": "decoder-only",
    "phi-3": "decoder-only",
    "pythia": "decoder-only",
    "stablelm": "decoder-only",
    "open-llama": "decoder-only",
    "openllama": "decoder-only",
    "qwen": "decoder-only",
    "qwen2": "decoder-only",
    
    # Encoder-decoder Models
    "t5": "encoder-decoder",
    "bart": "encoder-decoder",
    "pegasus": "encoder-decoder",
    "mbart": "encoder-decoder",
    "mt5": "encoder-decoder",
    "longt5": "encoder-decoder",
    "led": "encoder-decoder",
    "prophetnet": "encoder-decoder",
    "m2m_100": "encoder-decoder",
    "marian": "encoder-decoder",
    "opus-mt": "encoder-decoder",
    "flan-t5": "encoder-decoder",
    
    # Vision Models
    "vit": "vision",
    "deit": "vision",
    "beit": "vision",
    "swin": "vision",
    "resnet": "vision",
    "convnext": "vision",
    "bit": "vision",
    "dpt": "vision",
    "segformer": "vision",
    "detr": "vision",
    "yolos": "vision",
    "sam": "vision",
    "mask2former": "vision",
    "levit": "vision",
    "mlp-mixer": "vision",
    "mobilevit": "vision",
    "poolformer": "vision",
    "regnet": "vision",
    "efficientnet": "vision",
    "mobilenet_v1": "vision",
    "mobilenet_v2": "vision",
    "dinov2": "vision",
    
    # Vision-text Models
    "clip": "vision-encoder-text-decoder",
    "blip": "vision-encoder-text-decoder",
    "git": "vision-encoder-text-decoder",
    "align": "vision-encoder-text-decoder",
    "donut": "vision-encoder-text-decoder",
    "blip-2": "vision-encoder-text-decoder",
    "blip2": "vision-encoder-text-decoder",
    "vilt": "vision-encoder-text-decoder",
    "vinvl": "vision-encoder-text-decoder",
    
    # Speech Models
    "whisper": "speech",
    "wav2vec2": "speech",
    "hubert": "speech",
    "wavlm": "speech",
    "unispeech": "speech",
    "unispeech-sat": "speech",
    "sew": "speech",
    "sew-d": "speech",
    "data2vec": "speech",
    "encodec": "speech",
    "clap": "speech",
    "musicgen": "speech",
    "seamless_m4t": "speech",
    "usm": "speech",
    
    # Multimodal Models
    "llava": "multimodal",
    "flava": "multimodal",
    "flamingo": "multimodal",
    "idefics": "multimodal",
    "paligemma": "multimodal",
    "imagebind": "multimodal",
    "florence": "multimodal",
}

def normalize_model_name(model_name: str) -> str:
    """
    Normalize a model name to a standard format.
    
    Args:
        model_name: Model name
        
    Returns:
        Normalized model name
    """
    # Extract the base model name (remove organization)
    if "/" in model_name:
        model_name = model_name.split("/")[1]
    
    # Remove version numbers and sizes
    model_name = re.sub(r"-\d+b.*$", "", model_name.lower())
    model_name = re.sub(r"\.?v\d+.*$", "", model_name)
    
    # Handle common prefixes
    prefixes = ["hf-", "hf_", "huggingface-", "huggingface_"]
    for prefix in prefixes:
        if model_name.startswith(prefix):
            model_name = model_name[len(prefix):]
    
    # Remove common version suffixes
    suffixes = ["-base", "-small", "-large", "-tiny", "-mini", "-medium"]
    for suffix in suffixes:
        if model_name.endswith(suffix):
            model_name = model_name[:-len(suffix)]
    
    # Normalize hyphens
    model_name = model_name.replace("-", "_")
    
    return model_name

def get_architecture_type(model_name: str) -> str:
    """
    Get the architecture type for a model.
    
    Args:
        model_name: Model name
        
    Returns:
        Architecture type
    """
    # Normalize model name
    normalized_name = normalize_model_name(model_name)
    
    # Check direct mappings
    for name_pattern, arch_type in MODEL_NAME_MAPPING.items():
        if name_pattern in normalized_name:
            return arch_type
    
    # Check by regex patterns
    patterns = {
        "encoder-only": [r"bert", r"roberta", r"electra", r"deberta"],
        "decoder-only": [r"gpt", r"llama", r"mistral", r"falcon", r"phi"],
        "encoder-decoder": [r"t5", r"bart", r"pegasus", r"mbart"],
        "vision": [r"vit", r"swin", r"resnet", r"convnext"],
        "vision-encoder-text-decoder": [r"clip", r"blip"],
        "speech": [r"whisper", r"wav2vec", r"hubert", r"encodec"],
        "multimodal": [r"llava", r"flava", r"flamingo"]
    }
    
    for arch_type, pattern_list in patterns.items():
        for pattern in pattern_list:
            if re.search(pattern, normalized_name, re.IGNORECASE):
                return arch_type
    
    # Default to encoder-only if not found
    logger.warning(f"Could not determine architecture type for {model_name}, defaulting to encoder-only")
    return "encoder-only"

def get_model_metadata(model_name: str) -> Dict[str, Any]:
    """
    Get metadata for a model.
    
    Args:
        model_name: Model name
        
    Returns:
        Dictionary with model metadata
    """
    # Normalize model name
    normalized_name = normalize_model_name(model_name)
    
    # Determine architecture type
    arch_type = get_architecture_type(model_name)
    
    # Determine task based on architecture
    task_mapping = {
        "encoder-only": "fill-mask",
        "decoder-only": "text-generation",
        "encoder-decoder": "text2text-generation",
        "vision": "image-classification",
        "vision-encoder-text-decoder": "image-to-text",
        "speech": "automatic-speech-recognition",
        "multimodal": "multimodal-classification"
    }
    
    task = task_mapping.get(arch_type, "fill-mask")
    
    # Determine example input based on task
    example_input_mapping = {
        "fill-mask": '"The capital of France is [MASK]."',
        "text-generation": '"Once upon a time,"',
        "text2text-generation": '"translate English to French: Hello, how are you?"',
        "image-classification": 'Image.open("test.jpg")',
        "image-to-text": 'Image.open("test.jpg")',
        "automatic-speech-recognition": 'load_audio_file("test.mp3")',
        "multimodal-classification": '[Image.open("test.jpg"), "A photo of"]'
    }
    
    example_input = example_input_mapping.get(task, '"The capital of France is [MASK]."')
    
    # Return metadata
    return {
        "normalized_name": normalized_name,
        "architecture_type": arch_type,
        "task": task,
        "example_input": example_input
    }

if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) > 1:
        model_name = sys.argv[1]
        
        print(f"Analyzing model: {model_name}")
        print(f"Normalized name: {normalize_model_name(model_name)}")
        print(f"Architecture type: {get_architecture_type(model_name)}")
        print(f"Metadata: {get_model_metadata(model_name)}")
    else:
        print("Usage: python architecture_detector.py <model_name>")