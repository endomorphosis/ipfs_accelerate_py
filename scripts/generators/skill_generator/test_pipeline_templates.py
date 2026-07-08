#!/usr/bin/env python3
"""
Test script for the pipeline templates in the refactored generator suite.

This script tests the new vision-text and audio pipeline templates by generating
model implementations for various model architectures.
"""

import os
import sys
import logging
from typing import Dict, List, Any

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import templates
from templates.base_pipeline import BasePipelineTemplate
# Need to import the TextPipelineTemplate from the actual file
from templates.base_pipeline import TextPipelineTemplate
# The ImagePipelineTemplate is not yet defined, so we'll use a mock
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
        
    def get_mock_input_code(self):
        return "# Mock image input code"
        
    def get_mock_output_code(self):
        return "# Mock image output code"
        
    def is_compatible_with_architecture(self, arch_type):
        return arch_type in ["vision"]

from templates.vision_text_pipeline import VisionTextPipelineTemplate
from templates.audio_pipeline import AudioPipelineTemplate

# Import architecture templates
from templates.base_architecture import BaseArchitectureTemplate
from templates.encoder_only import EncoderOnlyArchitectureTemplate
from templates.decoder_only import DecoderOnlyArchitectureTemplate
from templates.encoder_decoder import EncoderDecoderArchitectureTemplate
from templates.vision import VisionArchitectureTemplate
from templates.vision_text import VisionTextArchitectureTemplate
from templates.speech import SpeechArchitectureTemplate

# Import hardware templates
from templates.base_hardware import BaseHardwareTemplate
from templates.cpu_hardware import CPUHardwareTemplate

# Import template composer
from templates.template_composer import TemplateComposer

# Define output directory
OUTPUT_DIR = "pipeline_test_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def test_pipeline_templates():
    """Test the pipeline templates with a variety of model architectures."""
    
    # Initialize pipeline templates
    pipeline_templates = {
        "text": TextPipelineTemplate(),
        "image": ImagePipelineTemplate(),
        "vision-text": VisionTextPipelineTemplate(),
        "audio": AudioPipelineTemplate()
    }
    
    # Initialize architecture templates
    architecture_templates = {
        "encoder-only": EncoderOnlyArchitectureTemplate(),
        "decoder-only": DecoderOnlyArchitectureTemplate(),
        "encoder-decoder": EncoderDecoderArchitectureTemplate(),
        "vision": VisionArchitectureTemplate(),
        "vision-encoder-text-decoder": VisionTextArchitectureTemplate(),
        "speech": SpeechArchitectureTemplate()
    }
    
    # Initialize hardware templates
    hardware_templates = {
        "cpu": CPUHardwareTemplate()
    }
    
    # Initialize template composer
    composer = TemplateComposer(hardware_templates, architecture_templates, pipeline_templates, OUTPUT_DIR)
    
    # Test models to generate
    test_models = [
        {"model_name": "bert", "arch_type": "encoder-only", "hardware_types": ["cpu"]},
        {"model_name": "gpt2", "arch_type": "decoder-only", "hardware_types": ["cpu"]},
        {"model_name": "t5-small", "arch_type": "encoder-decoder", "hardware_types": ["cpu"]},
        {"model_name": "vit", "arch_type": "vision", "hardware_types": ["cpu"]},
        {"model_name": "clip", "arch_type": "vision-encoder-text-decoder", "hardware_types": ["cpu"]},
        {"model_name": "whisper", "arch_type": "speech", "hardware_types": ["cpu"]}
    ]
    
    # Generate implementations for test models
    for model_config in test_models:
        model_name = model_config["model_name"]
        arch_type = model_config["arch_type"]
        hardware_types = model_config["hardware_types"]
        
        logger.info(f"Generating implementation for {model_name} ({arch_type})...")
        success, output_file = composer.generate_model_implementation(
            model_name, arch_type, hardware_types, force=True
        )
        
        if success:
            logger.info(f"Successfully generated implementation at {output_file}")
        else:
            logger.error(f"Failed to generate implementation for {model_name}")


def validate_pipeline_compatibility():
    """Test the pipeline compatibility with different architectures."""
    logger.info("Testing pipeline-architecture compatibility...")
    
    # Initialize pipeline templates
    pipeline_templates = {
        "text": TextPipelineTemplate(),
        "image": ImagePipelineTemplate(),
        "vision-text": VisionTextPipelineTemplate(),
        "audio": AudioPipelineTemplate()
    }
    
    # Architecture types
    architectures = [
        "encoder-only", "decoder-only", "encoder-decoder", 
        "vision", "vision-encoder-text-decoder", "speech"
    ]
    
    # Check compatibility
    for pipe_type, pipeline in pipeline_templates.items():
        logger.info(f"Testing {pipe_type} pipeline compatibility:")
        for arch in architectures:
            is_compatible = pipeline.is_compatible_with_architecture(arch)
            logger.info(f"  - {arch}: {'Compatible' if is_compatible else 'Not compatible'}")


def main():
    """Main function."""
    validate_pipeline_compatibility()
    test_pipeline_templates()


if __name__ == "__main__":
    main()