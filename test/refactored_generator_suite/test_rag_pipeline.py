#!/usr/bin/env python3
"""
Test script for Retrieval-Augmented Generation (RAG) pipeline and model templates.

This script tests the RAG pipeline and model templates with different task types
to ensure they generate the expected code and integrate properly with the
architecture template system.
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
from templates.rag import RAGArchitectureTemplate
from templates.rag_pipeline import RAGPipelineTemplate
# Import TestRagModel directly from the template file, which would fail due to template variables
# Instead, we'll test the file's contents directly
from templates.cpu_hardware import CPUHardwareTemplate
from templates.cuda_hardware import CudaHardwareTemplate
from templates.rocm_hardware import RocmHardwareTemplate
from templates.template_composer import TemplateComposer


def create_test_environment():
    """Create test environment with templates."""
    # Initialize templates
    rag_arch = RAGArchitectureTemplate()
    rag_pipeline = RAGPipelineTemplate()
    cpu_hardware = CPUHardwareTemplate()
    cuda_hardware = CudaHardwareTemplate()
    rocm_hardware = RocmHardwareTemplate()
    
    # Create template mappings
    architecture_templates = {'rag': rag_arch}
    pipeline_templates = {'rag': rag_pipeline}
    hardware_templates = {'cpu': cpu_hardware, 'cuda': cuda_hardware, 'rocm': rocm_hardware}
    
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


def test_rag_generative_qa():
    """Test generating a RAG model implementation for generative QA."""
    logger.info("Testing RAG generative QA implementation...")
    
    composer, output_dir = create_test_environment()
    
    # Generate model implementation
    # The file will be named based on the architecture type, not the model name
    success, output_file = composer.generate_model_implementation(
        model_name='rag-sequence',  # This is just for documentation
        arch_type='rag',            # This determines the filename
        hardware_types=['cpu', 'cuda'],
        force=True
    )
    
    if success:
        logger.info(f"Successfully generated RAG implementation at {output_file}")
        
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
        assert "class hf_rag:" in content, "Missing correct class name"
            
        # Check for RAG-specific components
        assert "Retrieval-Augmented Generation" in content, "Missing RAG reference"
        assert "num_docs" in content, "Missing num_docs parameter"
        assert "retrieval_context" in content, "Missing retrieval_context parameter"
        
        # Check for RAG-specific pipeline imports
        assert "# rag pipeline imports" in content or "# RAG pipeline imports" in content, "Missing RAG pipeline imports"
        
        # Check for task-specific components
        assert "generative_qa" in content, "Missing generative_qa task"
        
        # Check for utility functions
        assert "evaluate_document_relevance" in content, "Missing document relevance evaluation function"
        assert "format_context_from_documents" in content, "Missing context formatting function"
        
        logger.info("✅ RAG generative QA implementation test PASSED!")
        return True
    else:
        logger.error("❌ Failed to generate RAG implementation")
        return False


def test_rag_document_retrieval():
    """Test generating a RAG model for document retrieval."""
    logger.info("Testing RAG document retrieval implementation...")
    
    composer, output_dir = create_test_environment()
    
    # Generate model implementation with document_retrieval task type
    # We need to override the default task type
    composer.architecture_templates['rag'].default_task_type = "document_retrieval"
    
    # Note: This will overwrite the previous implementation file
    success, output_file = composer.generate_model_implementation(
        model_name='rag-token',          # Just for documentation
        arch_type='rag',                 # This determines the filename
        hardware_types=['cpu'],
        force=True
    )
    
    # Reset the default task type for other tests
    composer.architecture_templates['rag'].default_task_type = "generative_qa"
    
    if success:
        logger.info(f"Successfully generated RAG document retrieval implementation at {output_file}")
        
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
        assert "class hf_rag:" in content, "Missing correct class name"
        
        # Check for document retrieval-specific components
        assert "document_retrieval" in content, "Missing document_retrieval task"
        assert "retrieved_docs" in content, "Missing retrieved_docs in output processing"
        assert "query" in content, "Missing query parameter"
        
        logger.info("✅ RAG document retrieval implementation test PASSED!")
        return True
    else:
        logger.error("❌ Failed to generate RAG document retrieval implementation")
        return False


def verify_pipeline(task_type='generative_qa'):
    """Test RAG pipeline code generation for a specific task."""
    logger.info(f"Testing pipeline code generation for task: {task_type}")
    
    # Create templates
    rag_pipeline = RAGPipelineTemplate()
    
    # Generate preprocessing code
    preprocessing = rag_pipeline.get_preprocessing_code(task_type)
    assert preprocessing, f"Failed to generate preprocessing code for {task_type}"
    logger.info(f"✅ Generated preprocessing code for {task_type}")
    
    # Generate postprocessing code
    postprocessing = rag_pipeline.get_postprocessing_code(task_type)
    assert postprocessing, f"Failed to generate postprocessing code for {task_type}"
    logger.info(f"✅ Generated postprocessing code for {task_type}")
    
    # Generate result formatting code
    formatting = rag_pipeline.get_result_formatting_code(task_type)
    assert formatting, f"Failed to generate result formatting code for {task_type}"
    logger.info(f"✅ Generated result formatting code for {task_type}")
    
    # Check for RAG-specific features
    assert any(rag_term in preprocessing for rag_term in ["num_docs", "retrieval_context"]), \
        f"Missing RAG parameters in preprocessing for {task_type}"
    
    return True


def test_template_composer_integration():
    """Test that template composer properly maps RAG architecture."""
    logger.info("Testing template composer integration...")
    
    composer, output_dir = create_test_environment()
    
    # Check architecture to pipeline mapping
    arch_template = RAGArchitectureTemplate()
    cpu_hardware = CPUHardwareTemplate()
    
    # Get pipeline template through select_templates_for_model
    _, _, pipeline_template = composer.select_templates_for_model(
        model_name="rag-sequence",
        arch_type="rag",
        hardware_types=["cpu"]
    )
    
    # Check that the pipeline template is the RAG pipeline
    assert pipeline_template.pipeline_type == "rag", "Template composer did not map rag to RAG pipeline"
    
    logger.info("✅ Template composer integration test PASSED!")
    return True


def test_rag_model_template():
    """Test the RAG model template file."""
    logger.info("Testing RAG model template...")
    
    try:
        # Check template file structure
        template_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    'templates/rag_model_template.py')
        assert os.path.exists(template_path), f"Template file not found at {template_path}"
        
        # Read the template file to check its content
        with open(template_path, 'r') as f:
            content = f.read()
        
        # Verify model template contains essential RAG components
        assert "class {skillset_class_name}" in content, "Missing main class template"
        assert "initialize_retriever" in content, "Missing retriever initialization method"
        assert "retrieve_documents" in content, "Missing document retrieval method"
        assert "retrieve_and_generate" in content, "Missing retrieve and generate method"
        assert "analyze_retrieval_quality" in content, "Missing retrieval quality analysis"
        
        # Verify ROCm support
        assert "device == \"rocm\"" in content, "Missing ROCm hardware support"
        assert "AMD" in content and "Radeon" in content, "Missing AMD GPU detection"
        
        # Verify hardware awareness
        assert "supports_half_precision" in content, "Missing half-precision support detection"
        assert "hardware_info" in content, "Missing hardware info tracking"
        
        # Verify test infrastructure
        assert "class TestRagModel" in content, "Missing test class"
        assert "def run_tests" in content, "Missing test runner method"
        
        # Verify RAG-specific functionality
        assert "_create_mock_retriever" in content, "Missing mock retriever creation"
        assert "_load_documents_from_file" in content, "Missing document loading function"
        assert "chat" in content, "Missing chat functionality"
        assert "embed" in content, "Missing embedding functionality"
        
        # Verify specialized RAG methods
        assert "get_parameters" in content, "Missing parameters method"
        assert "format_context_from_documents" in content or "format_context" in content, "Missing context formatting"
        
        logger.info("✅ RAG model template test PASSED!")
        return True
        
    except Exception as e:
        logger.error(f"❌ RAG model template test failed: {str(e)}")
        return False


def run_all_tests():
    """Run all RAG pipeline tests."""
    logger.info("Starting RAG pipeline tests...")
    
    results = []
    
    # Test pipeline code generation for each task type
    for task_type in ['generative_qa', 'document_retrieval']:
        results.append(verify_pipeline(task_type))
    
    # Test template composer integration
    results.append(test_template_composer_integration())
    
    # Test the RAG model template
    results.append(test_rag_model_template())
    
    # Test full model implementation generation
    results.append(test_rag_generative_qa())
    results.append(test_rag_document_retrieval())
    
    # Report overall results
    if all(results):
        logger.info("🎉 All RAG pipeline tests PASSED! 🎉")
        return True
    else:
        logger.error(f"❌ Some tests failed! Passed: {sum(results)}/{len(results)}")
        return False


if __name__ == "__main__":
    run_all_tests()