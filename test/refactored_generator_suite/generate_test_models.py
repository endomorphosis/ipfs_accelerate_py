#!/usr/bin/env python3
"""
Test script to generate model implementations using our pipeline templates.

This script will generate complete model implementations for vision-text and speech models
to verify that our pipeline templates are working correctly.
"""

import os
import sys
import logging
import importlib
from typing import Dict, Any, List, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import architecture templates
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from templates.vision_text import VisionTextArchitectureTemplate
from templates.speech import SpeechArchitectureTemplate
from templates.multimodal import MultimodalArchitectureTemplate

# Import hardware templates
from templates.cpu_hardware import CPUHardwareTemplate
from templates.cuda_hardware import CudaHardwareTemplate

# Import pipeline templates
from templates.vision_text_pipeline import VisionTextPipelineTemplate
from templates.audio_pipeline import AudioPipelineTemplate
from templates.multimodal_pipeline import MultimodalPipelineTemplate

# Import template composer if it exists
try:
    from templates.template_composer import TemplateComposer
    COMPOSER_AVAILABLE = True
except ImportError:
    logger.warning("TemplateComposer not available, will use direct template composition")
    COMPOSER_AVAILABLE = False

def create_output_dir(output_dir='generated_models'):
    """Create output directory for generated models."""
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def generate_vision_text_model(output_dir, model_name='clip'):
    """Generate a vision-text model implementation."""
    logger.info(f"Generating vision-text model implementation for {model_name}...")
    
    try:
        # Initialize templates
        vision_text_arch = VisionTextArchitectureTemplate()
        cpu_hardware = CPUHardwareTemplate()
        vision_text_pipeline = VisionTextPipelineTemplate()
        
        # Generate file path
        file_path = os.path.join(output_dir, f"hf_{model_name}.py")
        
        if COMPOSER_AVAILABLE:
            # Use template composer
            hardware_templates = {'cpu': cpu_hardware}
            architecture_templates = {'vision-encoder-text-decoder': vision_text_arch}
            pipeline_templates = {'vision-text': vision_text_pipeline}
            
            composer = TemplateComposer(hardware_templates, architecture_templates, pipeline_templates, output_dir)
            success, output_file = composer.generate_model_implementation(
                model_name=model_name,
                arch_type='vision-encoder-text-decoder',
                hardware_types=['cpu'],
                force=True
            )
            
            if success:
                logger.info(f"Successfully generated vision-text model implementation at {output_file}")
                return output_file
            else:
                logger.error(f"Failed to generate vision-text model implementation")
                return None
        else:
            # Direct template composition
            logger.warning("Using direct template composition instead of TemplateComposer")
            # Basic implementation - in a real scenario, we would implement the direct composition logic
            with open(file_path, 'w') as f:
                f.write("# Generated vision-text model implementation\n")
                f.write("# This is a placeholder for direct template composition\n")
                f.write("# In a real scenario, we would combine the templates manually\n")
            logger.info(f"Generated placeholder implementation at {file_path}")
            return file_path
            
    except Exception as e:
        logger.error(f"Error generating vision-text model implementation: {e}")
        return None

def generate_speech_model(output_dir, model_name='whisper'):
    """Generate a speech model implementation."""
    logger.info(f"Generating speech model implementation for {model_name}...")
    
    try:
        # Initialize templates
        speech_arch = SpeechArchitectureTemplate()
        cpu_hardware = CPUHardwareTemplate()
        audio_pipeline = AudioPipelineTemplate()
        
        # Generate file path
        file_path = os.path.join(output_dir, f"hf_{model_name}.py")
        
        if COMPOSER_AVAILABLE:
            # Use template composer
            hardware_templates = {'cpu': cpu_hardware}
            architecture_templates = {'speech': speech_arch}
            pipeline_templates = {'audio': audio_pipeline}
            
            composer = TemplateComposer(hardware_templates, architecture_templates, pipeline_templates, output_dir)
            success, output_file = composer.generate_model_implementation(
                model_name=model_name,
                arch_type='speech',
                hardware_types=['cpu'],
                force=True
            )
            
            if success:
                logger.info(f"Successfully generated speech model implementation at {output_file}")
                return output_file
            else:
                logger.error(f"Failed to generate speech model implementation")
                return None
        else:
            # Direct template composition
            logger.warning("Using direct template composition instead of TemplateComposer")
            # Basic implementation - in a real scenario, we would implement the direct composition logic
            with open(file_path, 'w') as f:
                f.write("# Generated speech model implementation\n")
                f.write("# This is a placeholder for direct template composition\n")
                f.write("# In a real scenario, we would combine the templates manually\n")
            logger.info(f"Generated placeholder implementation at {file_path}")
            return file_path
            
    except Exception as e:
        logger.error(f"Error generating speech model implementation: {e}")
        return None

def generate_multimodal_model(output_dir, model_name='flava', use_cuda=False):
    """Generate a multimodal model implementation."""
    logger.info(f"Generating multimodal model implementation for {model_name}...")
    
    try:
        # Initialize templates
        multimodal_arch = MultimodalArchitectureTemplate()
        cpu_hardware = CPUHardwareTemplate()
        cuda_hardware = CudaHardwareTemplate()
        multimodal_pipeline = MultimodalPipelineTemplate()
        
        # Generate file path
        file_path = os.path.join(output_dir, f"hf_{model_name}.py")
        
        if COMPOSER_AVAILABLE:
            # Use template composer
            hardware_templates = {'cpu': cpu_hardware, 'cuda': cuda_hardware}
            architecture_templates = {'multimodal': multimodal_arch}
            pipeline_templates = {'multimodal': multimodal_pipeline}
            
            # Select hardware types based on use_cuda parameter
            hw_types = ['cpu', 'cuda'] if use_cuda else ['cpu']
            
            composer = TemplateComposer(hardware_templates, architecture_templates, pipeline_templates, output_dir)
            success, output_file = composer.generate_model_implementation(
                model_name=model_name,
                arch_type='multimodal',
                hardware_types=hw_types,
                force=True
            )
            
            if success:
                logger.info(f"Successfully generated multimodal model implementation at {output_file}")
                return output_file
            else:
                logger.error(f"Failed to generate multimodal model implementation")
                return None
        else:
            # Direct template composition
            logger.warning("Using direct template composition instead of TemplateComposer")
            # Basic implementation - in a real scenario, we would implement the direct composition logic
            with open(file_path, 'w') as f:
                f.write("# Generated multimodal model implementation\n")
                f.write("# This is a placeholder for direct template composition\n")
                f.write("# In a real scenario, we would combine the templates manually\n")
            logger.info(f"Generated placeholder implementation at {file_path}")
            return file_path
            
    except Exception as e:
        logger.error(f"Error generating multimodal model implementation: {e}")
        return None

def verify_generated_file(file_path):
    """Verify that the generated file has the expected content."""
    if not file_path or not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return False
    
    try:
        with open(file_path, 'r') as f:
            content = f.read()
            
        # Check for minimal expected content
        if len(content) < 100:
            logger.warning(f"File content is too short: {len(content)} bytes")
            return False
            
        logger.info(f"File {file_path} has valid content ({len(content)} bytes)")
        return True
    except Exception as e:
        logger.error(f"Error verifying file {file_path}: {e}")
        return False

def main():
    """Main function."""
    logger.info("Starting test model generation...")
    
    # Create output directory
    output_dir = create_output_dir('generated_test_models')
    
    # Generate vision-text model
    vision_text_file = generate_vision_text_model(output_dir, 'clip')
    verify_generated_file(vision_text_file)
    
    # Generate speech model
    speech_file = generate_speech_model(output_dir, 'whisper')
    verify_generated_file(speech_file)
    
    # Generate multimodal model (CPU only)
    multimodal_file = generate_multimodal_model(output_dir, 'flava', use_cuda=False)
    verify_generated_file(multimodal_file)
    
    # Generate multimodal model with CUDA support
    multimodal_cuda_file = generate_multimodal_model(output_dir, 'llava', use_cuda=True)
    verify_generated_file(multimodal_cuda_file)
    
    logger.info("Test model generation completed.")

if __name__ == "__main__":
    main()