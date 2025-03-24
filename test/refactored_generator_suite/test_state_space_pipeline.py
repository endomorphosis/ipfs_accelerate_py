#!/usr/bin/env python3
"""
Test script for State-Space pipeline template.

This script tests the State-Space pipeline template with different task types
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
from templates.state_space import StateSpaceArchitectureTemplate
from templates.state_space_pipeline import StateSpacePipelineTemplate
from templates.cpu_hardware import CPUHardwareTemplate
from templates.cuda_hardware import CudaHardwareTemplate
from templates.template_composer import TemplateComposer


def create_test_environment():
    """Create test environment with templates."""
    # Initialize templates
    state_space_arch = StateSpaceArchitectureTemplate()
    state_space_pipeline = StateSpacePipelineTemplate()
    cpu_hardware = CPUHardwareTemplate()
    cuda_hardware = CudaHardwareTemplate()
    
    # Create template mappings
    architecture_templates = {'state-space': state_space_arch}
    pipeline_templates = {'state-space': state_space_pipeline}
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


def test_state_space_generation():
    """Test generating a State-Space model implementation for text generation."""
    logger.info("Testing State-Space generation implementation...")
    
    composer, output_dir = create_test_environment()
    
    # Generate model implementation
    # The file will be named based on the architecture type, not the model name
    success, output_file = composer.generate_model_implementation(
        model_name='mamba-2-7b',  # This is just for documentation
        arch_type='state-space',   # This determines the filename
        hardware_types=['cpu', 'cuda'],
        force=True
    )
    
    if success:
        logger.info(f"Successfully generated State-Space implementation at {output_file}")
        
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
        assert "class hf_state_space:" in content, "Missing correct class name"
            
        # Check for State-Space-specific components
        assert "State-Space" in content, "Missing State-Space reference"
        assert "chunk_size" in content, "Missing chunk_size parameter"
        assert "state_decode" in content, "Missing state_decode parameter"
        
        # Check for State-Space-specific pipeline imports
        assert "# state-space pipeline imports" in content or "# State-Space pipeline imports" in content, "Missing State-Space pipeline imports"
        
        # Check for task-specific components
        assert "text_generation" in content, "Missing text_generation task"
        
        # Check for efficiency analysis functions
        assert "analyze_state_efficiency" in content, "Missing state efficiency analysis function"
        assert "estimate_memory_usage" in content, "Missing memory usage estimation function"
        
        logger.info("✅ State-Space implementation test PASSED!")
        return True
    else:
        logger.error("❌ Failed to generate State-Space implementation")
        return False


def test_state_space_classification():
    """Test generating a State-Space model for classification."""
    logger.info("Testing State-Space classification implementation...")
    
    composer, output_dir = create_test_environment()
    
    # Generate model implementation with text_classification task type
    # We need to override the default task type
    composer.architecture_templates['state-space'].default_task_type = "text_classification"
    
    # Note: This will overwrite the previous implementation file
    success, output_file = composer.generate_model_implementation(
        model_name='rwkv-4-raven',         # Just for documentation
        arch_type='state-space',            # This determines the filename
        hardware_types=['cpu'],
        force=True
    )
    
    # Reset the default task type for other tests
    composer.architecture_templates['state-space'].default_task_type = "text_generation"
    
    if success:
        logger.info(f"Successfully generated State-Space classification implementation at {output_file}")
        
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
        assert "class hf_state_space:" in content, "Missing correct class name"
        
        # Check for classification-specific components
        assert "text_classification" in content, "Missing text_classification task"
        assert "probabilities" in content, "Missing probabilities in output processing"
        assert "predicted_class_ids" in content, "Missing class prediction"
        
        # Check for State-Space-specific parameters
        assert "state_info" in content, "Missing state information"
        
        logger.info("✅ State-Space classification implementation test PASSED!")
        return True
    else:
        logger.error("❌ Failed to generate State-Space classification implementation")
        return False


def verify_pipeline(task_type='text_generation'):
    """Test State-Space pipeline code generation for a specific task."""
    logger.info(f"Testing pipeline code generation for task: {task_type}")
    
    # Create templates
    state_space_pipeline = StateSpacePipelineTemplate()
    
    # Generate preprocessing code
    preprocessing = state_space_pipeline.get_preprocessing_code(task_type)
    assert preprocessing, f"Failed to generate preprocessing code for {task_type}"
    logger.info(f"✅ Generated preprocessing code for {task_type}")
    
    # Generate postprocessing code
    postprocessing = state_space_pipeline.get_postprocessing_code(task_type)
    assert postprocessing, f"Failed to generate postprocessing code for {task_type}"
    logger.info(f"✅ Generated postprocessing code for {task_type}")
    
    # Generate result formatting code
    formatting = state_space_pipeline.get_result_formatting_code(task_type)
    assert formatting, f"Failed to generate result formatting code for {task_type}"
    logger.info(f"✅ Generated result formatting code for {task_type}")
    
    # Check for State-Space-specific features
    assert any(state_term in preprocessing for state_term in ["chunk_size", "state_decode"]), \
        f"Missing state-space parameters in preprocessing for {task_type}"
    
    return True


def test_template_composer_integration():
    """Test that template composer properly maps State-Space architecture."""
    logger.info("Testing template composer integration...")
    
    composer, output_dir = create_test_environment()
    
    # Check architecture to pipeline mapping
    arch_template = StateSpaceArchitectureTemplate()
    cpu_hardware = CPUHardwareTemplate()
    
    # Get pipeline template through select_templates_for_model
    _, _, pipeline_template = composer.select_templates_for_model(
        model_name="mamba-2-7b",
        arch_type="state-space",
        hardware_types=["cpu"]
    )
    
    # Check that the pipeline template is the State-Space pipeline
    assert pipeline_template.pipeline_type == "state-space", "Template composer did not map state-space to State-Space pipeline"
    
    logger.info("✅ Template composer integration test PASSED!")
    return True


def run_all_tests():
    """Run all State-Space pipeline tests."""
    logger.info("Starting State-Space pipeline tests...")
    
    results = []
    
    # Test pipeline code generation for each task type
    for task_type in ['text_generation', 'text_classification', 'feature_extraction']:
        results.append(verify_pipeline(task_type))
    
    # Test template composer integration
    results.append(test_template_composer_integration())
    
    # Test full model implementation generation
    results.append(test_state_space_generation())
    results.append(test_state_space_classification())
    
    # Report overall results
    if all(results):
        logger.info("🎉 All State-Space pipeline tests PASSED! 🎉")
        return True
    else:
        logger.error(f"❌ Some tests failed! Passed: {sum(results)}/{len(results)}")
        return False


if __name__ == "__main__":
    run_all_tests()