#!/usr/bin/env python3
"""
Create reference implementations using the modular template system.

This script uses the modular template system to generate reference implementations
for different model architectures and hardware backends.
"""

import os
import sys
import logging
import argparse
from typing import Dict, List, Any, Optional

# Import template classes
from templates.base_hardware import BaseHardwareTemplate, CPUHardwareTemplate
from templates.cuda_hardware import CudaHardwareTemplate
from templates.rocm_hardware import RocmHardwareTemplate
from templates.openvino_hardware import OpenvinoHardwareTemplate
from templates.apple_hardware import AppleHardwareTemplate
from templates.qualcomm_hardware import QualcommHardwareTemplate

from templates.base_architecture import BaseArchitectureTemplate
# Import existing architecture templates
from templates.encoder_only import EncoderOnlyArchitectureTemplate
from templates.decoder_only import DecoderOnlyArchitectureTemplate
from templates.encoder_decoder import EncoderDecoderArchitectureTemplate
from templates.vision import VisionArchitectureTemplate
from templates.speech import SpeechArchitectureTemplate
from templates.vision_text import VisionTextArchitectureTemplate
from templates.multimodal import MultimodalArchitectureTemplate

from templates.base_pipeline import BasePipelineTemplate, TextPipelineTemplate
from templates.image_pipeline import ImagePipelineTemplate
from templates.vision_text_pipeline import VisionTextPipelineTemplate
from templates.audio_pipeline import AudioPipelineTemplate
from templates.multimodal_pipeline import MultimodalPipelineTemplate

# Import the template composer
from templates.template_composer import TemplateComposer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_pipeline_templates() -> Dict[str, BasePipelineTemplate]:
    """
    Create pipeline templates for different input/output types.
    
    Returns:
        Dictionary mapping pipeline types to pipeline templates
    """
    return {
        "text": TextPipelineTemplate(),
        "image": ImagePipelineTemplate(),
        "vision-text": VisionTextPipelineTemplate(),
        "audio": AudioPipelineTemplate(),
        "multimodal": MultimodalPipelineTemplate()
        # Future: Add diffusion, moe, state-space, and rag pipeline templates
    }


def create_architecture_templates() -> Dict[str, BaseArchitectureTemplate]:
    """
    Create architecture templates for different model architectures.
    
    Returns:
        Dictionary mapping architecture types to architecture templates
    """
    return {
        "encoder-only": EncoderOnlyArchitectureTemplate(),
        "decoder-only": DecoderOnlyArchitectureTemplate(),
        "encoder-decoder": EncoderDecoderArchitectureTemplate(),
        "vision": VisionArchitectureTemplate(),
        "speech": SpeechArchitectureTemplate(),
        "vision-encoder-text-decoder": VisionTextArchitectureTemplate(),
        "multimodal": MultimodalArchitectureTemplate()
    }


def create_hardware_templates() -> Dict[str, BaseHardwareTemplate]:
    """
    Create hardware templates for different hardware backends.
    
    Returns:
        Dictionary mapping hardware types to hardware templates
    """
    return {
        "cpu": CPUHardwareTemplate(),
        "cuda": CudaHardwareTemplate(),
        "rocm": RocmHardwareTemplate(),
        "openvino": OpenvinoHardwareTemplate(),
        "mps": AppleHardwareTemplate(),
        "qnn": QualcommHardwareTemplate()
    }


def get_model_type_from_autoconfig(model_name: str) -> Optional[str]:
    """
    Attempt to get the exact model type using autoConfig from transformers.
    
    Args:
        model_name: The model name or path
        
    Returns:
        The model type or None if detection fails
    """
    try:
        # Try to import transformers
        import transformers
        from transformers import AutoConfig
        
        # Load the model config
        logger.info(f"Attempting to load model config for {model_name}")
        config = AutoConfig.from_pretrained(model_name)
        
        # Get the model type from the config
        model_type = getattr(config, "model_type", None)
        
        if model_type:
            logger.info(f"Detected model type: {model_type}")
            return model_type
        else:
            logger.warning(f"Could not detect model type from config for {model_name}")
            return None
    except Exception as e:
        logger.warning(f"Error detecting model type for {model_name}: {e}")
        return None

def get_arch_type_for_model(model_name: str) -> Dict[str, str]:
    """
    Get the architecture type and model type for a model name.
    
    Args:
        model_name: The model name
        
    Returns:
        Dictionary with architecture_type and model_type
    """
    # This is a simplified mapping - in a real implementation,
    # we would use the architecture detector to determine this
    
    # Extract base model name without organization prefix
    base_name = model_name.split("/")[-1].lower()
    
    # Initialize result dictionary
    result = {
        "architecture_type": "encoder-only",  # Default architecture type
        "model_type": base_name               # Default model type is the base name
    }
    
    # First check for exact model family matches
    if any(name in base_name for name in ["bert", "roberta", "deberta", "albert", "electra"]):
        result["architecture_type"] = "encoder-only"
        # Extract the model type (bert, roberta, etc.)
        for model_family in ["bert", "roberta", "deberta", "albert", "electra"]:
            if model_family in base_name:
                result["model_type"] = model_family
                break
    
    elif any(name in base_name for name in ["gpt", "llama", "bloom", "mistral", "phi"]):
        result["architecture_type"] = "decoder-only"
        # Extract the model type (gpt, llama, etc.)
        for model_family in ["gpt", "llama", "bloom", "mistral", "phi"]:
            if model_family in base_name:
                result["model_type"] = model_family
                break
    
    elif any(name in base_name for name in ["t5", "bart", "pegasus", "mbart", "mt5"]):
        result["architecture_type"] = "encoder-decoder"
        # Extract the model type (t5, bart, etc.)
        for model_family in ["t5", "bart", "pegasus", "mbart", "mt5"]:
            if model_family in base_name:
                result["model_type"] = model_family
                break
    
    elif any(name in base_name for name in ["vit", "deit", "beit", "swin", "convnext"]):
        result["architecture_type"] = "vision"
        # Extract the model type (vit, deit, etc.)
        for model_family in ["vit", "deit", "beit", "swin", "convnext"]:
            if model_family in base_name:
                result["model_type"] = model_family
                break
    
    elif any(name in base_name for name in ["clip", "blip"]):
        result["architecture_type"] = "vision-encoder-text-decoder"
        # Extract the model type (clip, blip)
        for model_family in ["clip", "blip"]:
            if model_family in base_name:
                result["model_type"] = model_family
                break
                
    elif any(name in base_name for name in ["flava", "llava", "paligemma", "idefics", "imagebind"]):
        result["architecture_type"] = "multimodal"
        # Extract the model type (flava, llava, paligemma, etc.)
        for model_family in ["flava", "llava", "paligemma", "idefics", "imagebind"]:
            if model_family in base_name:
                result["model_type"] = model_family
                break
    
    elif any(name in base_name for name in ["wav2vec", "whisper", "hubert", "encodec", "musicgen"]):
        result["architecture_type"] = "speech"
        # Extract the model type (wav2vec, whisper, etc.)
        for model_family in ["wav2vec", "whisper", "hubert", "encodec", "musicgen"]:
            if model_family in base_name:
                result["model_type"] = model_family
                break
    
    else:
        # Default to encoder-only if unknown
        logger.warning(f"Unknown model architecture for {model_name}, defaulting to encoder-only")
    
    return result


def main():
    """Generate reference implementations using the modular template system."""
    parser = argparse.ArgumentParser(description="Generate reference implementations using modular templates")
    parser.add_argument("--model", type=str, help="Model name to generate implementation for", required=True)
    parser.add_argument("--output-dir", type=str, default="../ipfs_accelerate_py/worker/skillset", help="Output directory for generated files")
    parser.add_argument("--hardware", type=str, nargs="+", default=["cpu", "cuda"], 
                        help="Hardware backends to include (cpu, cuda, rocm, openvino, mps, qnn)")
    parser.add_argument("--force", action="store_true", help="Force overwrite existing files")
    parser.add_argument("--detect-model-type", action="store_true", help="Attempt to detect exact model type using autoConfig")
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Ensure directory path is properly formed
    if args.output_dir != "../ipfs_accelerate_py/worker/skillset":
        logger.warning("Output directory is not '../ipfs_accelerate_py/worker/skillset', which is the standard location")
        logger.warning("Consider using --output-dir ../ipfs_accelerate_py/worker/skillset for consistency")
    
    # Create templates
    hardware_templates = create_hardware_templates()
    architecture_templates = create_architecture_templates()
    pipeline_templates = create_pipeline_templates()
    
    # Create template composer
    composer = TemplateComposer(
        hardware_templates=hardware_templates,
        architecture_templates=architecture_templates,
        pipeline_templates=pipeline_templates,
        output_dir=args.output_dir
    )
    
    # Get model info from our detector
    model_info = get_arch_type_for_model(args.model)
    arch_type = model_info["architecture_type"]
    model_type = model_info["model_type"]
    
    # Try to get more specific model type if detection is enabled
    if args.detect_model_type:
        detected_type = get_model_type_from_autoconfig(args.model)
        if detected_type:
            # Use the detected type instead
            model_type = detected_type
    
    logger.info(f"Model {args.model} detected as architecture type: {arch_type}")
    logger.info(f"Using model type '{model_type}' for file naming")
    
    # Generate implementation
    success, output_file = composer.generate_model_implementation(
        model_name=model_type,
        arch_type=arch_type,
        hardware_types=args.hardware,
        force=args.force
    )
    
    if success:
        logger.info(f"Successfully generated implementation for {args.model} at {output_file}")
    else:
        logger.error(f"Failed to generate implementation for {args.model}")
        sys.exit(1)


if __name__ == "__main__":
    main()