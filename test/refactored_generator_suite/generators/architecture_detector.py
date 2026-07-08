#!/usr/bin/env python3
"""
Architecture detector module for the refactored generator suite.

This module provides functions to detect the architecture type
of a model based on its name or other attributes.
"""

import re
import logging
from typing import Dict, List, Optional, Tuple, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# List of supported architecture types
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
    "rag",
    "text-to-image",
    "protein-folding",
    "video-processing"
]

# Direct mapping of model names to architecture types
MODEL_NAME_MAPPING = {
    # Encoder-only models
    "bert": "encoder-only",
    "roberta": "encoder-only",
    "electra": "encoder-only",
    "deberta": "encoder-only",
    "albert": "encoder-only",
    "bart-encoder": "encoder-only",
    "camembert": "encoder-only",
    "xlm-roberta": "encoder-only",
    "mpnet": "encoder-only",
    "ernie": "encoder-only",
    "distilbert": "encoder-only",
    "mobilebert": "encoder-only",
    "flaubert": "encoder-only",
    "layoutlm": "encoder-only",
    "canine": "encoder-only",
    "luke": "encoder-only",
    "roformer": "encoder-only",
    "convbert": "encoder-only",
    "funnel": "encoder-only",
    "rembert": "encoder-only",
    "herbert": "encoder-only",
    "tapas": "encoder-only",
    "markuplm": "encoder-only",
    "qdqbert": "encoder-only",
    "ibert": "encoder-only",
    "mega": "encoder-only",
    "data2vec-text": "encoder-only",
    "poolformer": "encoder-only",
    
    # Protein folding models (previously classified as encoder-only)
    "esm": "protein-folding",
    "esm1": "protein-folding",
    "esm2": "protein-folding",
    "esm-1b": "protein-folding",
    "esm-2": "protein-folding",
    "prot-bert": "protein-folding",
    "prot-bert-base": "protein-folding",
    "prot-t5": "protein-folding",
    "proteinbert": "protein-folding",
    "proteint5": "protein-folding",
    "esmfold": "protein-folding",
    "esmif": "protein-folding",
    "openfold": "protein-folding",
    "alphafold": "protein-folding",

    # Decoder-only models
    "gpt": "decoder-only",
    "gpt2": "decoder-only",
    "gpt-2": "decoder-only",
    "gpt-neo": "decoder-only",
    "gptneo": "decoder-only",
    "gpt-j": "decoder-only",
    "gptj": "decoder-only",
    "gpt-neox": "decoder-only",
    "gptneox": "decoder-only",
    "bloom": "decoder-only",
    "bloomz": "decoder-only",
    "opt": "decoder-only",
    "llama": "decoder-only",
    "llama2": "decoder-only",
    "llama-2": "decoder-only",
    "llama3": "decoder-only",
    "llama-3": "decoder-only",
    "mistral": "decoder-only",
    "falcon": "decoder-only",
    "phi": "decoder-only",
    "phi-1": "decoder-only",
    "phi-1.5": "decoder-only",
    "phi-2": "decoder-only",
    "phi-3": "decoder-only",
    "pythia": "decoder-only",
    "cerebras": "decoder-only",
    "codegen": "decoder-only",
    "santacoder": "decoder-only",
    "biogpt": "decoder-only",
    "ctrl": "decoder-only",
    "dolly": "decoder-only",
    "cohere": "decoder-only",
    "qwen": "decoder-only",
    "qwen2": "decoder-only",
    "baichuan": "decoder-only",
    "xglm": "decoder-only",
    "cpm": "decoder-only",
    "cpmant": "decoder-only",
    "galactica": "decoder-only",
    "stablenlm": "decoder-only",
    "stablelm": "decoder-only",
    "olmo": "decoder-only",
    "gemma": "decoder-only",
    "mpt": "decoder-only",
    "jais": "decoder-only",

    # Encoder-decoder models
    "t5": "encoder-decoder",
    "mt5": "encoder-decoder",
    "bart": "encoder-decoder",
    "mbart": "encoder-decoder",
    "pegasus": "encoder-decoder",
    "longt5": "encoder-decoder",
    "led": "encoder-decoder",
    "prop": "encoder-decoder",
    "prophetnet": "encoder-decoder",
    "blenderbot": "encoder-decoder",
    "marian": "encoder-decoder",
    "opus-mt": "encoder-decoder",
    "m2m": "encoder-decoder",
    "m2m100": "encoder-decoder",
    "m2m-100": "encoder-decoder",
    "fsmt": "encoder-decoder",
    "bigbird-pegasus": "encoder-decoder",
    "reformer": "encoder-decoder",
    "flan-t5": "encoder-decoder",
    "flan-ul2": "encoder-decoder",
    "flant5": "encoder-decoder",
    "umt5": "encoder-decoder",
    "mt0": "encoder-decoder",
    "ulmt5": "encoder-decoder",
    "nllb": "encoder-decoder",
    "pegasus-x": "encoder-decoder",
    "plbart": "encoder-decoder",
    "seamless-m4t": "encoder-decoder",
    "seamlessm4t": "encoder-decoder",
    "xlm-prophetnet": "encoder-decoder",

    # Vision models
    "vit": "vision",
    "deit": "vision",
    "beit": "vision",
    "convnext": "vision",
    "resnet": "vision",
    "swin": "vision",
    "bit": "vision",
    "mobilenet": "vision",
    "efficientnet": "vision",
    "regnet": "vision",
    "dino": "vision",
    "dinov2": "vision",
    "detr": "vision",
    "conditional-detr": "vision",
    "segformer": "vision",
    "dpt": "vision",
    "yolos": "vision",
    "mask2former": "vision",
    "levit": "vision",
    "vitdet": "vision",
    "vitmsn": "vision",
    "vitmatte": "vision",
    "data2vec-vision": "vision",
    "mobilevit": "vision",
    "focalnet": "vision",
    "nat": "vision",
    "pvt": "vision",
    "dinat": "vision",
    "swiftformer": "vision",
    "efficientformer": "vision",
    "deta": "vision",
    "dit": "vision",
    "mlp-mixer": "vision",
    "van": "vision",
    "ssd": "vision",

    # Vision-encoder-text-decoder models
    "clip": "vision-encoder-text-decoder",
    "alt-clip": "vision-encoder-text-decoder",
    "siglip": "vision-encoder-text-decoder",
    "blip": "vision-encoder-text-decoder",
    "blip-2": "vision-encoder-text-decoder",
    "blip2": "vision-encoder-text-decoder",
    "git": "vision-encoder-text-decoder",
    "pix2struct": "vision-encoder-text-decoder",
    "donut": "vision-encoder-text-decoder",
    "bridgetower": "vision-encoder-text-decoder",
    "trocr": "vision-encoder-text-decoder",
    "xclip": "vision-encoder-text-decoder",
    "chinese-clip": "vision-encoder-text-decoder",
    "clipseg": "vision-encoder-text-decoder",
    "groupvit": "vision-encoder-text-decoder",
    "align": "vision-encoder-text-decoder",
    "albef": "vision-encoder-text-decoder",
    "instructblip": "vision-encoder-text-decoder",
    "owlvit": "vision-encoder-text-decoder",
    "vilt": "vision-encoder-text-decoder",
    "kosmos2": "vision-encoder-text-decoder",
    "lxmert": "vision-encoder-text-decoder",
    "xvlm": "vision-encoder-text-decoder",
    "visual-bert": "vision-encoder-text-decoder",
    "vinvl": "vision-encoder-text-decoder",

    # Speech models
    "wav2vec2": "speech",
    "whisper": "speech",
    "hubert": "speech",
    "speech-to-text": "speech",
    "speecht5": "speech",
    "mms": "speech",
    "sew": "speech",
    "sew-d": "speech",
    "unispeech": "speech",
    "unispeech-sat": "speech",
    "mctct": "speech",
    "data2vec-audio": "speech",
    "wavlm": "speech",
    "xlsr-wav2vec2": "speech",
    "clap": "speech",
    "encodec": "speech",
    "mbart-large-50-many-to-many-mmt": "speech",
    "xls-r": "speech",
    "xlsr": "speech",

    # Multimodal models
    "flava": "multimodal",
    "llava": "multimodal",
    "fuyu": "multimodal",
    "idefics": "multimodal",
    "imagebind": "multimodal",
    "flamingo": "multimodal",
    "mplug-owl": "multimodal",
    "mplug-owl2": "multimodal",
    "paligemma": "multimodal",
    "musicgen": "multimodal",
    "florence": "multimodal",
    "gte": "multimodal",
    "clip-vision-model": "multimodal",
    "clvp": "multimodal",
    
    # Video processing models (previously classified as multimodal)
    "videomae": "video-processing",
    "vivit": "video-processing",
    "vivit-b-16x2": "video-processing",
    "vivit-b-16": "video-processing",
    "timesformer": "video-processing",
    "video-llama": "video-processing",
    "videollava": "video-processing",
    "video-gpt": "video-processing",
    "videobert": "video-processing",
    "tvc": "video-processing",
    "video-classifier": "video-processing",
    "video-captioning": "video-processing",
    "video-retrieval": "video-processing",
    "movie-clips": "video-processing",
    "movieclips": "video-processing",
    "action-recognition": "video-processing",
    "i3d": "video-processing",
    "c3d": "video-processing",
    "slowfast": "video-processing",
    "r3d": "video-processing",
    "x3d": "video-processing",

    # Text-to-image models (previously classified as diffusion)
    "stable-diffusion": "text-to-image",
    "latent-diffusion": "text-to-image",
    "sdxl": "text-to-image",
    "kandinsky": "text-to-image",
    "pixart": "text-to-image",
    "imagen": "text-to-image",
    "dalle": "text-to-image",
    "dalle-mini": "text-to-image", 
    "dalle2": "text-to-image",
    "dalle3": "text-to-image",
    "sd-turbo": "text-to-image",
    "midjourney": "text-to-image",
    "deepfloyd-if": "text-to-image",
    "controlnet": "text-to-image",
    
    # Segmentation models
    "sam": "diffusion",

    # Mixture of experts models
    "mixtral": "mixture-of-experts",
    "mixtral-8x7b": "mixture-of-experts",
    "mixtral-8x22b": "mixture-of-experts",
    "xmoe": "mixture-of-experts",
    "qwen-moe": "mixture-of-experts",
    "switch-transformers": "mixture-of-experts",
    "switchht": "mixture-of-experts",
    "olmoe": "mixture-of-experts",

    # State-space models
    "mamba": "state-space",
    "mamba-2": "state-space",
    "rwkv": "state-space",
    "s4": "state-space",
    "hyena": "state-space",

    # RAG models
    "rag": "rag",
    "rag-token": "rag",
    "rag-sequence": "rag",
    "rag-document": "rag"
}

# Patterns for model types when direct mapping doesn't work
MODEL_REGEX_PATTERNS = {
    "encoder-only": [
        r"bert",
        r"roberta",
        r"electra",
        r"deberta",
        r"canine",
        r"ernie",
    ],
    "decoder-only": [
        r"gpt",
        r"llama",
        r"falcon",
        r"phi",
        r"mistral",
        r"bloom",
        r"opt",
    ],
    "encoder-decoder": [
        r"t5",
        r"bart",
        r"pegasus",
        r"marian",
        r"m2m",
        r"seamless",
    ],
    "vision": [
        r"vit",
        r"swin",
        r"detr",
        r"dino",
        r"efficientnet",
        r"resnet",
    ],
    "vision-encoder-text-decoder": [
        r"clip",
        r"blip",
        r"kosmos",
        r"git[-_]",
    ],
    "speech": [
        r"wav2vec",
        r"whisper",
        r"hubert",
        r"speecht5",
        r"encodec",
    ],
    "multimodal": [
        r"flava",
        r"llava",
        r"mmpt",
        r"multiway",
    ],
    "diffusion": [
        r"segment[-_]anything",
        r"mask2former",
        r"segmentation",
    ],
    "mixture-of-experts": [
        r"mixtral",
        r"moe",
        r"switch",
    ],
    "state-space": [
        r"mamba",
        r"rwkv",
        r"s4",
        r"hyena",
    ],
    "rag": [
        r"rag",
        r"retriev",
    ],
    "text-to-image": [
        r"diffusion",
        r"stable[-_]",
        r"kandinsky",
        r"dalle",
        r"sdxl",
        r"imagen",
        r"txt2img",
        r"text[-_]to[-_]image",
    ],
    "protein-folding": [
        r"esm",
        r"prot[-_]bert",
        r"prot[-_]t5",
        r"fold",
        r"protein",
        r"structure[-_]prediction",
        r"alphafold",
    ],
    "video-processing": [
        r"video",
        r"timesformer",
        r"vivit",
        r"i3d",
        r"c3d",
        r"r3d",
        r"x3d",
        r"slowfast",
        r"action[-_]recognition",
        r"google/vivit",
    ]
}


def normalize_model_name(model_name: str) -> str:
    """
    Normalize a model name for architecture detection.
    
    Args:
        model_name: The model name or HF model ID
        
    Returns:
        Normalized model name
    """
    # Convert to lowercase
    model_name = model_name.lower()
    
    # Remove organization prefix if exists
    if "/" in model_name:
        model_name = model_name.split("/")[-1]
    
    # Remove version numbers or suffixes
    model_name = re.sub(r"[-_]v\d+", "", model_name)
    model_name = re.sub(r"[-_]\d+b", "", model_name)
    
    return model_name


def get_architecture_type(model_name: str) -> str:
    """
    Get the architecture type for a model based on its name.
    
    Args:
        model_name: The model name or Hugging Face model ID
        
    Returns:
        The architecture type
    """
    normalized_name = normalize_model_name(model_name)
    
    # Special cases for specific models
    if "google/vivit" in model_name.lower():
        return "video-processing"
        
    if "prot-bert" in model_name.lower():
        return "protein-folding"
    
    # First try direct mapping
    for name_pattern, arch_type in MODEL_NAME_MAPPING.items():
        if name_pattern in normalized_name:
            logger.debug(f"Matched {normalized_name} to {arch_type} via direct mapping")
            return arch_type
    
    # Then try regex patterns
    for arch_type, patterns in MODEL_REGEX_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, normalized_name):
                logger.debug(f"Matched {normalized_name} to {arch_type} via regex pattern {pattern}")
                return arch_type
    
    # Default to encoder-only as a fallback
    logger.warning(f"Could not determine architecture type for {model_name}, defaulting to encoder-only")
    return "encoder-only"


def get_model_class_name(model_name: str, arch_type: str) -> Tuple[str, str]:
    """
    Get the model class name for a given model and architecture type.
    
    Args:
        model_name: The model name
        arch_type: The architecture type
        
    Returns:
        Tuple of (model_class_name, processor_class_name)
    """
    # Mapping of architecture types to class name patterns
    arch_to_class = {
        "encoder-only": ("AutoModelForMaskedLM", "AutoTokenizer"),
        "decoder-only": ("AutoModelForCausalLM", "AutoTokenizer"),
        "encoder-decoder": ("AutoModelForSeq2SeqLM", "AutoTokenizer"),
        "vision": ("AutoModelForImageClassification", "AutoImageProcessor"),
        "vision-encoder-text-decoder": ("AutoModel", "AutoProcessor"),
        "speech": ("AutoModelForSpeechSeq2Seq", "AutoProcessor"),
        "multimodal": ("AutoModelForVision2Seq", "AutoProcessor"),
        "diffusion": ("AutoModel", "AutoProcessor"),
        "mixture-of-experts": ("AutoModelForCausalLM", "AutoTokenizer"),
        "state-space": ("AutoModelForCausalLM", "AutoTokenizer"),
        "rag": ("RagModel", "RagTokenizer"),
        "text-to-image": ("StableDiffusionPipeline", "CLIPTextProcessor"),
        "protein-folding": ("EsmModel", "AutoTokenizer"),
        "video-processing": ("VideoMAEForVideoClassification", "VideoMAEImageProcessor")
    }
    
    # Special cases based on specific model names
    normalized_name = normalize_model_name(model_name)
    
    # Vision-encoder-text-decoder models
    if arch_type == "vision-encoder-text-decoder":
        if "clip" in normalized_name:
            return "CLIPModel", "CLIPProcessor"
        elif "blip" in normalized_name:
            return "BlipForConditionalGeneration", "BlipProcessor"
        elif "git" in normalized_name:
            return "GitForCausalLM", "GitProcessor"
    # Speech models
    elif arch_type == "speech" and "whisper" in normalized_name:
        return "WhisperForConditionalGeneration", "WhisperProcessor"
    # Multimodal models
    elif arch_type == "multimodal" and "llava" in normalized_name:
        return "LlavaForConditionalGeneration", "LlavaProcessor"
    # Text-to-image models
    elif arch_type == "text-to-image":
        if "dalle" in normalized_name:
            return "DallePipeline", "DalleProcessor"
        elif "kandinsky" in normalized_name:
            return "KandinskyPipeline", "KandinskyProcessor"
        elif "sdxl" in normalized_name or "xl" in normalized_name:
            return "StableDiffusionXLPipeline", "CLIPTextProcessor"
        else:
            return "StableDiffusionPipeline", "CLIPTextProcessor"
    # Protein folding models
    elif arch_type == "protein-folding":
        if "fold" in normalized_name:
            return "EsmFoldModel", "EsmTokenizer"
        else:
            return "EsmModel", "EsmTokenizer"
    # Video processing models
    elif arch_type == "video-processing":
        if "vivit" in normalized_name:
            return "VivitForVideoClassification", "VivitImageProcessor"
        elif "timesformer" in normalized_name:
            return "TimesformerForVideoClassification", "TimesformerImageProcessor"
        else:
            return "VideoMAEForVideoClassification", "VideoMAEImageProcessor"
    
    # Default based on architecture type
    model_class, processor_class = arch_to_class.get(arch_type, ("AutoModel", "AutoTokenizer"))
    return model_class, processor_class


def get_model_class_name_short(model_class_name: str) -> str:
    """
    Get the short model class name (for use with frameworks like OpenVINO).
    
    Args:
        model_class_name: The full model class name
        
    Returns:
        Short model class name
    """
    # Remove "Auto" prefix and common suffixes
    short_name = model_class_name
    
    if short_name.startswith("Auto"):
        short_name = short_name[4:]
    
    # Special handling for specific model classes that don't follow the pattern
    if model_class_name == "CLIPModel":
        return "VisionText"
    elif model_class_name == "BlipForConditionalGeneration":
        return "VisionText"
    elif model_class_name == "GitForCausalLM":
        return "VisionText"
    elif model_class_name == "WhisperForConditionalGeneration":
        return "Speech"
    elif model_class_name == "LlavaForConditionalGeneration":
        return "VisionText"
    elif model_class_name == "RagModel":
        return "Rag"
    # New architecture types
    elif model_class_name == "StableDiffusionPipeline" or model_class_name == "DallePipeline" or model_class_name == "KandinskyPipeline":
        return "TextToImage"
    elif model_class_name == "StableDiffusionXLPipeline":
        return "TextToImage"
    elif model_class_name == "EsmModel" or model_class_name == "EsmFoldModel":
        return "ProteinFolding"
    elif model_class_name == "VideoMAEForVideoClassification" or model_class_name == "VivitForVideoClassification" or model_class_name == "TimesformerForVideoClassification":
        return "VideoProcessing"
    
    return short_name


def get_default_model_id(model_name: str, arch_type: str) -> str:
    """
    Get a default model ID for a given model type and architecture.
    
    Args:
        model_name: The model name
        arch_type: The architecture type
        
    Returns:
        Default model ID
    """
    # If model_name is already a full model ID (has a slash), use it
    if "/" in model_name:
        return model_name
    
    # Default model IDs by architecture type
    defaults = {
        "encoder-only": "bert-base-uncased",
        "decoder-only": "gpt2",
        "encoder-decoder": "t5-small",
        "vision": "google/vit-base-patch16-224",
        "vision-encoder-text-decoder": "openai/clip-vit-base-patch32",
        "speech": "openai/whisper-tiny",
        "multimodal": "facebook/flava-full",
        "diffusion": "facebook/sam-vit-base",
        "mixture-of-experts": "mistralai/Mixtral-8x7B-v0.1",
        "state-space": "state-spaces/mamba-2.8b",
        "rag": "facebook/rag-token-nq",
        "text-to-image": "runwayml/stable-diffusion-v1-5",
        "protein-folding": "facebook/esm2_t33_650M_UR50D",
        "video-processing": "MCG-NJU/videomae-base-finetuned-kinetics"
    }
    
    # Special case mappings based on model name
    normalized_name = normalize_model_name(model_name)
    
    if arch_type == "encoder-only":
        if "roberta" in normalized_name:
            return "roberta-base"
        elif "distilbert" in normalized_name:
            return "distilbert-base-uncased"
        elif "albert" in normalized_name:
            return "albert-base-v2"
    elif arch_type == "decoder-only":
        if "llama" in normalized_name:
            return "meta-llama/Llama-2-7b-hf"
        elif "mistral" in normalized_name:
            return "mistralai/Mistral-7B-v0.1"
        elif "phi" in normalized_name:
            return "microsoft/phi-2"
    elif arch_type == "vision-encoder-text-decoder":
        if "clip" in normalized_name:
            return "openai/clip-vit-base-patch32"
        elif "blip" in normalized_name:
            return "Salesforce/blip-image-captioning-base"
    elif arch_type == "speech":
        if "whisper" in normalized_name:
            return "openai/whisper-tiny"
        elif "wav2vec2" in normalized_name:
            return "facebook/wav2vec2-base-960h"
    elif arch_type == "text-to-image":
        if "sdxl" in normalized_name:
            return "stabilityai/stable-diffusion-xl-base-1.0"
        elif "kandinsky" in normalized_name:
            return "kandinsky-community/kandinsky-2-1"
        elif "dalle" in normalized_name:
            return "dall-e/mini"
    elif arch_type == "protein-folding":
        if "esm1" in normalized_name:
            return "facebook/esm1b_t33_650M_UR50S"
        elif "esm2" in normalized_name:
            return "facebook/esm2_t33_650M_UR50D"
        elif "fold" in normalized_name:
            return "facebook/esmfold_v1"
    elif arch_type == "video-processing":
        if "vivit" in normalized_name:
            return "google/vivit-b-16x2"
        elif "timesformer" in normalized_name:
            return "facebook/timesformer-base-finetuned-k400"
    
    # Return the default for the architecture type
    return defaults.get(arch_type, model_name)


def get_model_metadata(model_name: str) -> Dict[str, Any]:
    """
    Get metadata for a model based on its name.
    
    Args:
        model_name: The model name or HF model ID
        
    Returns:
        A dictionary of model metadata
    """
    # Get architecture type
    arch_type = get_architecture_type(model_name)
    
    # Get model class name and processor class name
    model_class_name, processor_class_name = get_model_class_name(model_name, arch_type)
    
    # Get short model class name
    model_class_name_short = get_model_class_name_short(model_class_name)
    
    # Get default model ID
    default_model_id = get_default_model_id(model_name, arch_type)
    
    # Determine model type from model name
    normalized_name = normalize_model_name(model_name)
    if "/" in model_name:
        model_type = normalized_name.split("-")[0] if "-" in normalized_name else normalized_name
    else:
        model_type = normalized_name
    
    # Create metadata
    metadata = {
        "model_name": model_name,
        "architecture_type": arch_type,
        "model_class_name": model_class_name,
        "processor_class_name": processor_class_name,
        "model_class_name_short": model_class_name_short,
        "default_model_id": default_model_id,
        "model_type": model_type
    }
    
    return metadata


# For testing
if __name__ == "__main__":
    # Test with various model names
    test_models = [
        # Basic architecture types
        "bert-base-uncased",
        "roberta-base",
        "gpt2",
        "t5-small",
        "google/vit-base-patch16-224",
        "openai/clip-vit-base-patch32",
        "openai/whisper-tiny",
        "facebook/flava-full",
        "mistralai/Mixtral-8x7B-v0.1",
        "llama",
        "phi-2",
        "clip",
        "blip",
        
        # New architecture types
        # Text-to-image models
        "runwayml/stable-diffusion-v1-5",
        "stabilityai/stable-diffusion-xl-base-1.0",
        "kandinsky-community/kandinsky-2-1",
        "dalle",
        
        # Protein folding models
        "facebook/esm2_t33_650M_UR50D",
        "facebook/esm1b_t33_650M_UR50S",
        "facebook/esmfold_v1",
        "proteinbert",
        
        # Video processing models
        "MCG-NJU/videomae-base-finetuned-kinetics",
        "google/vivit-b-16x2",
        "facebook/timesformer-base-finetuned-k400",
        "video-classifier"
    ]
    
    for model in test_models:
        metadata = get_model_metadata(model)
        print(f"\nModel: {model}")
        for key, value in metadata.items():
            print(f"  {key}: {value}")