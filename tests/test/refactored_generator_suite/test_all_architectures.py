#!/usr/bin/env python3
"""
Test all architecture types with the template composer.

This script tests the ability to generate implementations for all supported
Hugging Face model architectures.
"""

import os
import sys
import logging
from typing import Dict, Any, List

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import necessary modules
from templates.template_composer import TemplateComposer
from templates.base_pipeline import TextPipelineTemplate, BasePipelineTemplate
from templates.cpu_hardware import CPUHardwareTemplate

# Import all architecture templates
from templates.encoder_only import EncoderOnlyArchitectureTemplate
from templates.decoder_only import DecoderOnlyArchitectureTemplate
from templates.encoder_decoder import EncoderDecoderArchitectureTemplate
from templates.vision import VisionArchitectureTemplate
from templates.vision_text import VisionTextArchitectureTemplate
from templates.speech import SpeechArchitectureTemplate
from templates.multimodal import MultimodalArchitectureTemplate
from templates.diffusion import DiffusionArchitectureTemplate
from templates.moe import MoEArchitectureTemplate
from templates.state_space import StateSpaceArchitectureTemplate
from templates.rag import RAGArchitectureTemplate

# Import or mock image pipeline
try:
    from templates.image_pipeline import ImagePipelineTemplate
except ImportError:
    # Create a simple mock
    class ImagePipelineTemplate(BasePipelineTemplate):
        """Temporary mock for image pipeline template."""
        def __init__(self):
            super().__init__()
            self.pipeline_type = "image"
            self.input_type = "image"
            self.output_type = "image"
            
        def get_import_statements(self):
            return "# Image pipeline imports"
            
        def get_preprocessing_code(self, task_type):
            return "# Image preprocessing code"
            
        def get_postprocessing_code(self, task_type):
            return "# Image postprocessing code"
            
        def get_result_formatting_code(self, task_type):
            return "# Image result formatting code"
            
        def is_compatible_with_architecture(self, arch_type):
            return arch_type in ["vision"]

# Import specialized pipeline templates
from templates.vision_text_pipeline import VisionTextPipelineTemplate
from templates.audio_pipeline import AudioPipelineTemplate
from templates.multimodal_pipeline import MultimodalPipelineTemplate
from templates.diffusion_pipeline import DiffusionPipelineTemplate
from templates.moe_pipeline import MoEPipelineTemplate
from templates.state_space_pipeline import StateSpacePipelineTemplate
from templates.rag_pipeline import RAGPipelineTemplate


def test_all_architectures():
    """Test the generation of all architecture types."""
    # Initialize templates
    architecture_templates = {
        "encoder-only": EncoderOnlyArchitectureTemplate(),
        "decoder-only": DecoderOnlyArchitectureTemplate(),
        "encoder-decoder": EncoderDecoderArchitectureTemplate(),
        "vision": VisionArchitectureTemplate(),
        "vision-encoder-text-decoder": VisionTextArchitectureTemplate(),
        "speech": SpeechArchitectureTemplate(),
        "multimodal": MultimodalArchitectureTemplate(),
        "diffusion": DiffusionArchitectureTemplate(),
        "mixture-of-experts": MoEArchitectureTemplate(),
        "state-space": StateSpaceArchitectureTemplate(),
        "rag": RAGArchitectureTemplate()
    }
    
    pipeline_templates = {
        "text": TextPipelineTemplate(),
        "image": ImagePipelineTemplate(),
        "vision-text": VisionTextPipelineTemplate(),
        "audio": AudioPipelineTemplate(),
        "multimodal": MultimodalPipelineTemplate(),
        "diffusion": DiffusionPipelineTemplate(),
        "moe": MoEPipelineTemplate(),
        "state-space": StateSpacePipelineTemplate(),
        "rag": RAGPipelineTemplate()
    }
    
    hardware_templates = {
        "cpu": CPUHardwareTemplate()
    }
    
    # Output directory for generated implementations
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_all_implementations')
    os.makedirs(output_dir, exist_ok=True)
    
    # Create template composer
    composer = TemplateComposer(
        hardware_templates=hardware_templates,
        architecture_templates=architecture_templates,
        pipeline_templates=pipeline_templates,
        output_dir=output_dir
    )
    
    # Model configurations to test
    model_configs = [
        {"name": "bert-base-uncased", "type": "encoder-only", "hardware": ["cpu"]},
        {"name": "gpt2", "type": "decoder-only", "hardware": ["cpu"]},
        {"name": "t5-small", "type": "encoder-decoder", "hardware": ["cpu"]},
        {"name": "vit-base-patch16-224", "type": "vision", "hardware": ["cpu"]},
        {"name": "clip-vit-base-patch16", "type": "vision-encoder-text-decoder", "hardware": ["cpu"]},
        {"name": "whisper-small", "type": "speech", "hardware": ["cpu"]},
        {"name": "flava-full", "type": "multimodal", "hardware": ["cpu"]},
        {"name": "stable-diffusion-v1-5", "type": "diffusion", "hardware": ["cpu"]},
        {"name": "mixtral-8x7b", "type": "mixture-of-experts", "hardware": ["cpu"]},
        {"name": "mamba-2.8b", "type": "state-space", "hardware": ["cpu"]},
        {"name": "rag-sequence", "type": "rag", "hardware": ["cpu"]}
    ]
    
    # Test generation for each model
    results = []
    for model in model_configs:
        name = model["name"]
        arch_type = model["type"]
        hardware = model["hardware"]
        
        logger.info(f"Generating implementation for {name} ({arch_type})...")
        
        success, output_file = composer.generate_model_implementation(
            model_name=name,
            arch_type=arch_type,
            hardware_types=hardware,
            force=True
        )
        
        result = {
            "model": name,
            "architecture": arch_type,
            "success": success,
            "output_file": output_file if success else None
        }
        
        results.append(result)
        
        if success:
            logger.info(f"✅ Successfully generated {arch_type} implementation: {output_file}")
            # Verify file size
            file_size = os.path.getsize(output_file)
            logger.info(f"   File size: {file_size} bytes")
            if file_size < 10000:
                logger.warning(f"   ⚠️ File is suspiciously small: {file_size} bytes")
        else:
            logger.error(f"❌ Failed to generate {arch_type} implementation")
    
    # Generate summary report
    success_count = sum(1 for r in results if r["success"])
    logger.info(f"\nSummary: {success_count}/{len(results)} implementations generated successfully")
    
    if success_count == len(results):
        logger.info("🎉 All architecture types generated successfully!")
        return True
    else:
        logger.error("❌ Some architecture types failed to generate.")
        # List failures
        for result in results:
            if not result["success"]:
                logger.error(f"   Failed: {result['model']} ({result['architecture']})")
        return False


if __name__ == "__main__":
    test_all_architectures()