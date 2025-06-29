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
    "multimodal",
    "diffusion",
    "mixture-of-experts",
    "state-space",
    "rag"
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
    "mt0": "encoder-decoder",
    "nllb": "encoder-decoder",
    
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
    
    # Diffusion Models
    "stable-diffusion": "diffusion",
    "sdxl": "diffusion",
    "ssd": "diffusion",
    "dalle": "diffusion",
    "midjourney": "diffusion",
    "imagen": "diffusion",
    "kandinsky": "diffusion",
    "pixart": "diffusion",
    "latent-diffusion": "diffusion",
    
    # Mixture-of-Experts Models
    "mixtral": "mixture-of-experts",
    "mixtral-8x7b": "mixture-of-experts",
    "mixtral-8x22b": "mixture-of-experts",
    "switchht": "mixture-of-experts",
    "switchc": "mixture-of-experts",
    "qwen-moe": "mixture-of-experts",
    "qwen_moe": "mixture-of-experts",
    "qwen2-moe": "mixture-of-experts",
    "qwen2_moe": "mixture-of-experts",
    "xmoe": "mixture-of-experts",
    "olmo-moe": "mixture-of-experts",
    "olmo_moe": "mixture-of-experts",
    "olmoe": "mixture-of-experts",
    
    # State Space Models
    "mamba": "state-space",
    "mamba-2": "state-space",
    "gamba": "state-space",
    "vim": "state-space",
    "hyena": "state-space",
    "rwkv": "state-space",
    "rwkv5": "state-space",
    "ssd": "state-space",
    
    # RAG Models
    "rag": "rag",
    "rag-token": "rag",
    "rag-sequence": "rag",
    "rag-end2end": "rag",
    "rag-document": "rag"
}

def normalize_model_name(model_name: str) -> str:
    """
    Normalize a model name to a standard format.
    
    Args:
        model_name: Model name
        
    Returns:
        Normalized model name
    """
    # Save the original name for MoE special case
    original_name = model_name
    
    # Extract the base model name (remove organization)
    if "/" in model_name:
        model_name = model_name.split("/")[1]
    
    # Remove version numbers and sizes (but be careful with MoE patterns)
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
    
    # Special case for MoE models: keep the -moe or _moe suffix for detection
    if "-moe" in original_name.lower() or "_moe" in original_name.lower():
        # Extract just the base part and add _moe for consistent handling
        base_part = re.sub(r"[\-_]moe.*$", "", model_name.lower())
        return f"{base_part}_moe"
    
    # Normalize hyphens and periods (replace with underscore)
    model_name = model_name.replace("-", "_")
    model_name = model_name.replace(".", "_")
    
    return model_name

def get_architecture_type(model_name: str) -> str:
    """
    Get the architecture type for a model.
    
    Args:
        model_name: Model name
        
    Returns:
        Architecture type
    """
    # Special case for qwen-moe (this is a direct fix for a specific issue)
    if "qwen-moe" in model_name.lower() or "qwen_moe" in model_name.lower():
        return "mixture-of-experts"
    
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
        "multimodal": [r"llava", r"flava", r"flamingo"],
        "diffusion": [r"diffusion", r"stable-diffusion", r"sdxl", r"dalle", r"imagen", r"kandinsky", r"pixart", r"latent-diffusion"],
        "mixture-of-experts": [r"mixtral", r"switchht", r"switchc", r"-moe", r"_moe", r"qwen-moe", r"qwen_moe", r"olmo-moe", r"olmo_moe", r"olmoe"],
        "state-space": [r"mamba", r"hyena", r"rwkv", r"vim", r"gamba", r"s4", r"s5", r"ssm"],
        "rag": [r"rag", r"rag-token", r"rag-sequence", r"rag-document"]
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
        "multimodal": "multimodal-classification",
        "diffusion": "text-to-image",
        "mixture-of-experts": "text-generation",
        "state-space": "text-generation",
        "rag": "retrieval-augmented-generation"
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
        "multimodal-classification": '[Image.open("test.jpg"), "A photo of"]',
        "text-to-image": '"A photo of a cat sitting on a beach at sunset"',
        "retrieval-augmented-generation": '{"query": "What is the capital of France?", "documents": ["Paris is the capital of France."]}'
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