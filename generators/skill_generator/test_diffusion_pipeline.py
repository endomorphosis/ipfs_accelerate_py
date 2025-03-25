#!/usr/bin/env python3
"""
Test script for diffusion pipeline template.

This script tests the diffusion pipeline template with different task types
to ensure it generates the expected code.
"""

import os
import sys
import logging
from typing import Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import templates
from templates.diffusion import DiffusionArchitectureTemplate
from templates.diffusion_pipeline import DiffusionPipelineTemplate
from templates.cpu_hardware import CPUHardwareTemplate
from templates.cuda_hardware import CudaHardwareTemplate
from templates.template_composer import TemplateComposer


def create_test_environment():
    """Create test environment with templates."""
    # Initialize templates
    diffusion_arch = DiffusionArchitectureTemplate()
    diffusion_pipeline = DiffusionPipelineTemplate()
    cpu_hardware = CPUHardwareTemplate()
    cuda_hardware = CudaHardwareTemplate()
    
    # Create template mappings
    architecture_templates = {'diffusion': diffusion_arch}
    pipeline_templates = {'diffusion': diffusion_pipeline}
    hardware_templates = {'cpu': cpu_hardware, 'cuda': cuda_hardware}
    
    # Create output directory
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'generated_test_models')
    os.makedirs(output_dir, exist_ok=True)
    
    # Create template composer
    composer = TemplateComposer(
        hardware_templates=hardware_templates,
        architecture_templates=architecture_templates,
        pipeline_templates=pipeline_templates,
        output_dir=output_dir
    )
    
    return composer, output_dir


def test_stable_diffusion_generation():
    """Test generating a Stable Diffusion model implementation."""
    logger.info("Testing Stable Diffusion implementation generation...")
    
    composer, output_dir = create_test_environment()
    
    # Generate model implementation
    success, output_file = composer.generate_model_implementation(
        model_name='stable-diffusion',
        arch_type='diffusion',
        hardware_types=['cpu', 'cuda'],
        force=True
    )
    
    if success:
        logger.info(f"Successfully generated Stable Diffusion implementation at {output_file}")
        
        # Verify file was created
        assert os.path.exists(output_file), f"Output file {output_file} does not exist!"
        
        # Verify file size is reasonable
        file_size = os.path.getsize(output_file)
        logger.info(f"File size: {file_size} bytes")
        assert file_size > 10000, f"File size {file_size} is too small!"
        
        # Check for key components in the file
        with open(output_file, 'r') as f:
            content = f.read()
            
        # Check for diffusion-specific components
        assert "DiffusionPipeline" in content, "Missing DiffusionPipeline import"
        assert "guidance_scale" in content, "Missing guidance_scale parameter"
        assert "num_inference_steps" in content, "Missing num_inference_steps parameter"
        
        # Check for pipeline type detection
        assert "pipeline_type = \"diffusion\"" in content or "'diffusion'" in content, "Missing diffusion pipeline type"
        
        # Check for task-specific components
        assert "image_generation" in content, "Missing image_generation task"
        
        logger.info("✅ Stable Diffusion implementation test PASSED!")
        return True
    else:
        logger.error("❌ Failed to generate Stable Diffusion implementation")
        return False


def test_sam_segmentation():
    """Test generating a SAM model implementation."""
    logger.info("Testing SAM implementation generation...")
    
    composer, output_dir = create_test_environment()
    
    # Generate model implementation
    success, output_file = composer.generate_model_implementation(
        model_name='sam',
        arch_type='diffusion',
        hardware_types=['cpu'],
        force=True
    )
    
    if success:
        logger.info(f"Successfully generated SAM implementation at {output_file}")
        
        # Verify file was created
        assert os.path.exists(output_file), f"Output file {output_file} does not exist!"
        
        # Verify file size is reasonable
        file_size = os.path.getsize(output_file)
        logger.info(f"File size: {file_size} bytes")
        assert file_size > 10000, f"File size {file_size} is too small!"
        
        # Check for key components in the file
        with open(output_file, 'r') as f:
            content = f.read()
            
        # Check for segmentation-specific components
        assert "image_segmentation" in content, "Missing image_segmentation task"
        assert "input_points" in content, "Missing input_points parameter"
        assert "masks" in content, "Missing masks in output processing"
        
        logger.info("✅ SAM implementation test PASSED!")
        return True
    else:
        logger.error("❌ Failed to generate SAM implementation")
        return False


def test_inpainting_model():
    """Test generating an inpainting model implementation."""
    logger.info("Testing inpainting model implementation generation...")
    
    composer, output_dir = create_test_environment()
    
    # Generate model implementation
    success, output_file = composer.generate_model_implementation(
        model_name='stable-diffusion-inpainting',
        arch_type='diffusion',
        hardware_types=['cpu'],
        force=True
    )
    
    if success:
        logger.info(f"Successfully generated inpainting model implementation at {output_file}")
        
        # Verify file was created
        assert os.path.exists(output_file), f"Output file {output_file} does not exist!"
        
        # Verify file size is reasonable
        file_size = os.path.getsize(output_file)
        logger.info(f"File size: {file_size} bytes")
        assert file_size > 10000, f"File size {file_size} is too small!"
        
        # Check for key components in the file
        with open(output_file, 'r') as f:
            content = f.read()
            
        # Check for inpainting-specific components
        assert "inpainting" in content.lower(), "Missing inpainting task"
        assert "mask" in content.lower(), "Missing mask parameter"
        
        logger.info("✅ Inpainting model implementation test PASSED!")
        return True
    else:
        logger.error("❌ Failed to generate inpainting model implementation")
        return False


def verify_pipeline(task_type='image_generation'):
    """Test diffusion pipeline code generation for a specific task."""
    logger.info(f"Testing pipeline code generation for task: {task_type}")
    
    # Create templates
    diffusion_pipeline = DiffusionPipelineTemplate()
    
    # Generate preprocessing code
    preprocessing = diffusion_pipeline.get_preprocessing_code(task_type)
    assert preprocessing, f"Failed to generate preprocessing code for {task_type}"
    logger.info(f"✅ Generated preprocessing code for {task_type}")
    
    # Generate postprocessing code
    postprocessing = diffusion_pipeline.get_postprocessing_code(task_type)
    assert postprocessing, f"Failed to generate postprocessing code for {task_type}"
    logger.info(f"✅ Generated postprocessing code for {task_type}")
    
    # Generate result formatting code
    formatting = diffusion_pipeline.get_result_formatting_code(task_type)
    assert formatting, f"Failed to generate result formatting code for {task_type}"
    logger.info(f"✅ Generated result formatting code for {task_type}")
    
    return True


def run_all_tests():
    """Run all diffusion pipeline tests."""
    logger.info("Starting diffusion pipeline tests...")
    
    results = []
    
    # Test pipeline code generation for each task type
    for task_type in ['image_generation', 'image_to_image', 'inpainting', 'image_segmentation']:
        results.append(verify_pipeline(task_type))
    
    # Test full model implementation generation
    results.append(test_stable_diffusion_generation())
    results.append(test_sam_segmentation())
    results.append(test_inpainting_model())
    
    # Report overall results
    if all(results):
        logger.info("🎉 All diffusion pipeline tests PASSED! 🎉")
        return True
    else:
        logger.error(f"❌ Some tests failed! Passed: {sum(results)}/{len(results)}")
        return False


if __name__ == "__main__":
    run_all_tests()