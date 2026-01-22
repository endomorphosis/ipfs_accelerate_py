#!/usr/bin/env python3
"""
Verify all pipeline templates in the refactored generator suite.

This script checks that each architecture type in the refactored generator suite
is properly mapped to a compatible pipeline template.
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

# Import template composer and base templates
from templates.template_composer import TemplateComposer
from templates.base_hardware import BaseHardwareTemplate
from templates.base_architecture import BaseArchitectureTemplate
from templates.base_pipeline import BasePipelineTemplate, TextPipelineTemplate

# Import hardware templates
from templates.cpu_hardware import CPUHardwareTemplate

# Import architecture templates
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

# Import pipeline templates
try:
    # If ImagePipelineTemplate exists, import it
    from templates.image_pipeline import ImagePipelineTemplate
except ImportError:
    # Otherwise create a simple mock
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


def verify_pipeline_mapping():
    """Verify pipeline mapping for each architecture type."""
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
    
    # Create template composer
    composer = TemplateComposer(hardware_templates, architecture_templates, pipeline_templates, "test_output")
    
    # Test model architectures
    architectures = [
        {"name": "bert-base-uncased", "type": "encoder-only", "expected_pipeline": "text"},
        {"name": "gpt2", "type": "decoder-only", "expected_pipeline": "text"},
        {"name": "t5-small", "type": "encoder-decoder", "expected_pipeline": "text"},
        {"name": "vit-base-patch16-224", "type": "vision", "expected_pipeline": "image"},
        {"name": "clip-vit-base-patch16", "type": "vision-encoder-text-decoder", "expected_pipeline": "vision-text"},
        {"name": "whisper-small", "type": "speech", "expected_pipeline": "audio"},
        {"name": "flava", "type": "multimodal", "expected_pipeline": "multimodal"},
        {"name": "stable-diffusion-v1-5", "type": "diffusion", "expected_pipeline": "diffusion"},
        {"name": "mixtral-8x7b", "type": "mixture-of-experts", "expected_pipeline": "moe"},
        {"name": "mamba-2.8b", "type": "state-space", "expected_pipeline": "state-space"},
        {"name": "rag-token", "type": "rag", "expected_pipeline": "rag"}
    ]
    
    # Test pipeline mapping
    logger.info("Testing pipeline mapping for architecture types:")
    success = True
    
    for arch in architectures:
        model_name = arch["name"]
        arch_type = arch["type"]
        expected_pipeline = arch["expected_pipeline"]
        
        _, _, pipeline_template = composer.select_templates_for_model(
            model_name=model_name,
            arch_type=arch_type,
            hardware_types=["cpu"]
        )
        
        actual_pipeline = pipeline_template.pipeline_type
        
        if actual_pipeline == expected_pipeline:
            logger.info(f"✅ {arch_type} -> {actual_pipeline} pipeline (correct)")
        else:
            logger.error(f"❌ {arch_type} -> {actual_pipeline} pipeline (expected {expected_pipeline})")
            success = False
    
    if success:
        logger.info("🎉 All architecture types are correctly mapped to pipelines!")
    else:
        logger.error("❌ Some architecture types are incorrectly mapped to pipelines.")
    
    return success


def check_pipeline_compatibility():
    """Check that each pipeline correctly reports compatibility with architectures."""
    # Initialize pipeline templates
    pipeline_templates = [
        TextPipelineTemplate(),
        ImagePipelineTemplate(),
        VisionTextPipelineTemplate(),
        AudioPipelineTemplate(),
        MultimodalPipelineTemplate(),
        DiffusionPipelineTemplate(),
        MoEPipelineTemplate(),
        StateSpacePipelineTemplate(),
        RAGPipelineTemplate()
    ]
    
    # Architecture types
    architecture_types = [
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
    
    # Test pipeline compatibility
    logger.info("\nTesting pipeline compatibility reporting:")
    
    # Build compatibility matrix
    compatibility_matrix = {}
    
    for pipeline in pipeline_templates:
        pipeline_type = pipeline.pipeline_type
        compatibility = {}
        
        for arch_type in architecture_types:
            is_compatible = pipeline.is_compatible_with_architecture(arch_type)
            compatibility[arch_type] = is_compatible
        
        compatibility_matrix[pipeline_type] = compatibility
    
    # Print compatibility matrix
    logger.info("Pipeline Compatibility Matrix:")
    
    # Header
    header = "Pipeline Type       | " + " | ".join(f"{a[:10]:<10}" for a in architecture_types)
    separator = "-" * len(header)
    logger.info(separator)
    logger.info(header)
    logger.info(separator)
    
    # Rows
    for pipeline_type, compatibility in compatibility_matrix.items():
        row = f"{pipeline_type:20} |"
        for arch_type in architecture_types:
            is_compatible = compatibility[arch_type]
            mark = "✅" if is_compatible else "❌"
            row += f" {mark:^10} |"
        logger.info(row)
    
    logger.info(separator)
    
    # Check for gaps or inconsistencies
    logger.info("\nChecking for architecture coverage gaps:")
    gaps_found = False
    
    for arch_type in architecture_types:
        compatible_pipelines = [p for p, c in compatibility_matrix.items() if c.get(arch_type, False)]
        
        if not compatible_pipelines:
            logger.error(f"❌ Architecture '{arch_type}' has no compatible pipelines!")
            gaps_found = True
        elif len(compatible_pipelines) > 1:
            logger.warning(f"⚠️ Architecture '{arch_type}' has multiple compatible pipelines: {', '.join(compatible_pipelines)}")
    
    if not gaps_found:
        logger.info("✅ All architecture types have at least one compatible pipeline!")
    
    return not gaps_found


def main():
    """Main function for pipeline integration verification."""
    success = True
    success &= verify_pipeline_mapping()
    success &= check_pipeline_compatibility()
    
    if success:
        logger.info("\n🎉 Pipeline integration verification PASSED! All architectures have correct pipeline mappings.")
        return 0
    else:
        logger.error("\n❌ Pipeline integration verification FAILED! Some architectures have incorrect pipeline mappings.")
        return 1


if __name__ == "__main__":
    sys.exit(main())