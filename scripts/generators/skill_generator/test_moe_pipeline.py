#!/usr/bin/env python3
"""
Test script for Mixture-of-Experts (MoE) pipeline template.

This script tests the MoE pipeline template with different task types
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
from templates.moe import MoEArchitectureTemplate
from templates.moe_pipeline import MoEPipelineTemplate
from templates.cpu_hardware import CPUHardwareTemplate
from templates.cuda_hardware import CudaHardwareTemplate
from templates.template_composer import TemplateComposer


def create_test_environment():
    """Create test environment with templates."""
    # Initialize templates
    moe_arch = MoEArchitectureTemplate()
    moe_pipeline = MoEPipelineTemplate()
    cpu_hardware = CPUHardwareTemplate()
    cuda_hardware = CudaHardwareTemplate()
    
    # Create template mappings
    architecture_templates = {'mixture-of-experts': moe_arch}
    pipeline_templates = {'moe': moe_pipeline}
    hardware_templates = {'cpu': cpu_hardware, 'cuda': cuda_hardware}
    
    # Create output directory for HuggingFace transformers hardware backend implementations
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'transformers_implementations')
    os.makedirs(output_dir, exist_ok=True)
    
    # Create template composer
    composer = TemplateComposer(
        hardware_templates=hardware_templates,
        architecture_templates=architecture_templates,
        pipeline_templates=pipeline_templates,
        output_dir=output_dir
    )
    
    return composer, output_dir


def test_moe_generation():
    """Test generating a Mixture-of-Experts model implementation."""
    logger.info("Testing MoE implementation generation...")
    
    composer, output_dir = create_test_environment()
    
    # Generate model implementation
    # The file will be named based on the architecture type, not the model name
    success, output_file = composer.generate_model_implementation(
        model_name='mixtral-8x7b-instruct',  # This is just for documentation
        arch_type='mixture-of-experts',      # This determines the filename
        hardware_types=['cpu', 'cuda'],
        force=True
    )
    
    if success:
        logger.info(f"Successfully generated MoE implementation at {output_file}")
        
        # Verify file was created
        assert os.path.exists(output_file), f"Output file {output_file} does not exist!"
        
        # Verify file size is reasonable
        file_size = os.path.getsize(output_file)
        logger.info(f"File size: {file_size} bytes")
        assert file_size > 10000, f"File size {file_size} is too small!"
        
        # Check for key components in the file
        with open(output_file, 'r') as f:
            content = f.read()
        
        # Verify the class name is based on architecture type, not model name
        assert "class hf_mixture_of_experts:" in content, "Missing correct class name"
            
        # Check for MoE-specific components
        assert "Mixture-of-Experts" in content, "Missing Mixture-of-Experts reference"
        assert "num_experts_per_token" in content or "num_active_experts" in content, "Missing experts parameter"
        assert "expert_routing" in content, "Missing expert routing parameter"
        
        # Check for MoE-specific pipeline imports instead of pipeline_type variable
        assert "# moe pipeline imports" in content or "# MoE pipeline imports" in content, "Missing MoE pipeline imports"
        
        # Check for task-specific components
        assert "text_generation" in content, "Missing text_generation task"
        
        # Check for expert analysis functions
        assert "analyze_expert_usage" in content, "Missing expert analysis function"
        assert "extract_expert_patterns" in content, "Missing expert patterns function"
        
        logger.info("✅ MoE implementation test PASSED!")
        return True
    else:
        logger.error("❌ Failed to generate MoE implementation")
        return False


def test_moe_classification():
    """Test generating a Mixture-of-Experts model for classification."""
    logger.info("Testing MoE classification implementation generation...")
    
    composer, output_dir = create_test_environment()
    
    # Generate model implementation with text_classification task type
    # We need to override the default task type
    composer.architecture_templates['mixture-of-experts'].default_task_type = "text_classification"
    
    # Note: Since we're using the same architecture type, this will 
    # overwrite the previous implementation file
    success, output_file = composer.generate_model_implementation(
        model_name='switch-base-8',         # Just for documentation
        arch_type='mixture-of-experts',     # This determines the filename
        hardware_types=['cpu'],
        force=True
    )
    
    # Reset the default task type for other tests
    composer.architecture_templates['mixture-of-experts'].default_task_type = "text_generation"
    
    if success:
        logger.info(f"Successfully generated MoE classification implementation at {output_file}")
        
        # Verify file was created
        assert os.path.exists(output_file), f"Output file {output_file} does not exist!"
        
        # Verify file size is reasonable
        file_size = os.path.getsize(output_file)
        logger.info(f"File size: {file_size} bytes")
        assert file_size > 10000, f"File size {file_size} is too small!"
        
        # Check for key components in the file
        with open(output_file, 'r') as f:
            content = f.read()
            
        # Verify the class name is based on architecture type, not model name
        assert "class hf_mixture_of_experts:" in content, "Missing correct class name"
        
        # Check for classification-specific components
        assert "text_classification" in content, "Missing text_classification task"
        assert "probabilities" in content, "Missing probabilities in output processing"
        assert "predicted_class_ids" in content, "Missing class prediction"
        
        # Check for MoE expert routing information
        assert "expert_usage" in content, "Missing expert usage information"
        assert "router_logits" in content or "router_probs" in content, "Missing router probability information"
        
        logger.info("✅ MoE classification implementation test PASSED!")
        return True
    else:
        logger.error("❌ Failed to generate MoE classification implementation")
        return False


def verify_pipeline(task_type='text_generation'):
    """Test MoE pipeline code generation for a specific task."""
    logger.info(f"Testing pipeline code generation for task: {task_type}")
    
    # Create templates
    moe_pipeline = MoEPipelineTemplate()
    
    # Generate preprocessing code
    preprocessing = moe_pipeline.get_preprocessing_code(task_type)
    assert preprocessing, f"Failed to generate preprocessing code for {task_type}"
    logger.info(f"✅ Generated preprocessing code for {task_type}")
    
    # Generate postprocessing code
    postprocessing = moe_pipeline.get_postprocessing_code(task_type)
    assert postprocessing, f"Failed to generate postprocessing code for {task_type}"
    logger.info(f"✅ Generated postprocessing code for {task_type}")
    
    # Generate result formatting code
    formatting = moe_pipeline.get_result_formatting_code(task_type)
    assert formatting, f"Failed to generate result formatting code for {task_type}"
    logger.info(f"✅ Generated result formatting code for {task_type}")
    
    # Check for MoE-specific features
    assert any(expert_term in preprocessing for expert_term in ["num_active_experts", "expert_routing", "num_experts"]), \
        f"Missing expert parameters in preprocessing for {task_type}"
    
    return True


def test_template_composer_integration():
    """Test that template composer properly maps MoE architecture."""
    logger.info("Testing template composer integration...")
    
    composer, output_dir = create_test_environment()
    
    # Check architecture to pipeline mapping
    arch_template = MoEArchitectureTemplate()
    cpu_hardware = CPUHardwareTemplate()
    
    # Get pipeline template through select_templates_for_model
    _, _, pipeline_template = composer.select_templates_for_model(
        model_name="mixtral-8x7b",
        arch_type="mixture-of-experts",
        hardware_types=["cpu"]
    )
    
    # Check that the pipeline template is the MoE pipeline
    assert pipeline_template.pipeline_type == "moe", "Template composer did not map mixture-of-experts to MoE pipeline"
    
    logger.info("✅ Template composer integration test PASSED!")
    return True


def run_all_tests():
    """Run all MoE pipeline tests."""
    logger.info("Starting MoE pipeline tests...")
    
    results = []
    
    # Test pipeline code generation for each task type
    for task_type in ['text_generation', 'text_classification', 'feature_extraction']:
        results.append(verify_pipeline(task_type))
    
    # Test template composer integration
    results.append(test_template_composer_integration())
    
    # Test full model implementation generation
    results.append(test_moe_generation())
    results.append(test_moe_classification())
    
    # Report overall results
    if all(results):
        logger.info("🎉 All MoE pipeline tests PASSED! 🎉")
        return True
    else:
        logger.error(f"❌ Some tests failed! Passed: {sum(results)}/{len(results)}")
        return False


if __name__ == "__main__":
    run_all_tests()