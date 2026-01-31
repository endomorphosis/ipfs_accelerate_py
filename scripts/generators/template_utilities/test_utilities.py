#!/usr/bin/env python3
"""
Test script for template_utilities package.

This script tests the basic functionality of the template_utilities package
to ensure it works as expected.
"""

import os
import sys
import json
import logging
import tempfile
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import template_utilities modules
from template_utilities.placeholder_helpers import (
    get_standard_placeholders,
    extract_placeholders,
    detect_missing_placeholders,
    get_default_context,
    render_template,
    get_modality_for_model_type
)
from template_utilities.template_validation import (
    validate_template_syntax,
    validate_hardware_support,
    validate_template
)
from template_utilities.template_inheritance import (
    get_parent_for_model_type,
    get_inheritance_hierarchy,
    get_default_parent_templates
)
from template_utilities.template_database import (
    get_db_connection,
    create_schema,
    check_database,
    get_template,
    store_template,
    list_templates,
    add_default_parent_templates,
    update_template_inheritance,
    validate_all_templates
)

def test_placeholder_helpers():
    """Test placeholder helper functions"""
    logger.info("Testing placeholder helpers...")
    
    # Test get_standard_placeholders
    placeholders = get_standard_placeholders()
    assert len(placeholders) > 0, "No standard placeholders found"
    assert "model_name" in placeholders, "Required placeholder 'model_name' not found"
    
    # Test extract_placeholders
    template = """
    def test_{normalized_name}():
        model = load_model("{model_name}")
        device = "{torch_device}"
        has_cuda = {has_cuda}
    """
    extracted = extract_placeholders(template)
    assert "model_name" in extracted, "Failed to extract 'model_name' placeholder"
    assert "normalized_name" in extracted, "Failed to extract 'normalized_name' placeholder"
    assert "torch_device" in extracted, "Failed to extract 'torch_device' placeholder"
    assert "has_cuda" in extracted, "Failed to extract 'has_cuda' placeholder"
    
    # Test detect_missing_placeholders
    context = {
        "model_name": "bert-base-uncased",
        "normalized_name": "Bert_Base_Uncased",
        "torch_device": "cuda"
    }
    missing = detect_missing_placeholders(template, context)
    assert "has_cuda" in missing, "Failed to detect missing 'has_cuda' placeholder"
    
    # Test get_default_context
    context = get_default_context("bert-base-uncased")
    assert context["model_name"] == "bert-base-uncased", "Incorrect model_name in context"
    assert "normalized_name" in context, "Missing normalized_name in context"
    assert "torch_device" in context, "Missing torch_device in context"
    
    # Test render_template
    rendered = render_template(template, context)
    assert "bert-base-uncased" in rendered, "Failed to render model_name placeholder"
    assert context["normalized_name"] in rendered, "Failed to render normalized_name placeholder"
    
    # Test get_modality_for_model_type
    modality = get_modality_for_model_type("bert")
    assert modality == "text", "Incorrect modality for 'bert'"
    modality = get_modality_for_model_type("vit")
    assert modality == "vision", "Incorrect modality for 'vit'"
    modality = get_modality_for_model_type("whisper")
    assert modality == "audio", "Incorrect modality for 'whisper'"
    modality = get_modality_for_model_type("clip")
    assert modality == "multimodal", "Incorrect modality for 'clip'"
    
    logger.info("Placeholder helpers test passed!")

def test_template_validation():
    """Test template validation functions"""
    logger.info("Testing template validation...")
    
    # Test validate_template_syntax with valid template
    valid_template = """
    def test_function():
        model_name = "{model_name}"
        normalized_name = "{normalized_name}"
        return model_name
    """
    success, errors = validate_template_syntax(valid_template)
    assert success, f"Valid template failed syntax validation: {errors}"
    
    # Test validate_template_syntax with invalid template
    invalid_template = """
    def test_function():
        model_name = "{model_name"  # Missing closing brace
        return model_name
    """
    success, errors = validate_template_syntax(invalid_template)
    assert not success, "Invalid template passed syntax validation"
    
    # Test validate_hardware_support
    cuda_template = """
    import torch
    
    def test_function():
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
        return device
    """
    success, hardware_support = validate_hardware_support(cuda_template)
    assert hardware_support["cuda"], "Failed to detect CUDA support in template"
    
    # Test validate_template
    success, results = validate_template(valid_template, "test", "bert")
    assert "syntax" in results, "Missing syntax results in validation"
    assert "hardware" in results, "Missing hardware results in validation"
    assert "placeholders" in results, "Missing placeholders results in validation"
    
    logger.info("Template validation test passed!")

def test_template_inheritance():
    """Test template inheritance functions"""
    logger.info("Testing template inheritance...")
    
    # Test get_parent_for_model_type
    parent, modality = get_parent_for_model_type("bert")
    assert parent == "default_text", f"Incorrect parent for 'bert': {parent}"
    assert modality == "text", f"Incorrect modality for 'bert': {modality}"
    
    # Test get_inheritance_hierarchy
    hierarchy = get_inheritance_hierarchy("bert")
    assert hierarchy[0] == "bert", "First item in hierarchy should be the model type"
    assert hierarchy[1] == "default_text", "Second item should be the parent template"
    
    # Test get_default_parent_templates
    default_parents = get_default_parent_templates()
    assert "default_text" in default_parents, "Missing default_text in default parents"
    assert "default_vision" in default_parents, "Missing default_vision in default parents"
    assert "default_audio" in default_parents, "Missing default_audio in default parents"
    assert "default_multimodal" in default_parents, "Missing default_multimodal in default parents"
    
    logger.info("Template inheritance test passed!")

def test_template_database():
    """Test template database functions"""
    logger.info("Testing template database...")
    
    # Create a temporary database file
    with tempfile.NamedTemporaryFile(suffix=".duckdb", delete=False) as temp_db:
        db_path = temp_db.name
    
    try:
        # Test get_db_connection and create_schema
        conn = get_db_connection(db_path)
        assert conn is not None, "Failed to get database connection"
        
        success = create_schema(conn)
        assert success, "Failed to create database schema"
        conn.close()
        
        # Test check_database
        success = check_database(db_path)
        assert success, "Database check failed"
        
        # Test store_template
        test_template = """
        def test_{normalized_name}():
            model = load_model("{model_name}")
            device = "{torch_device}"
        """
        success = store_template(
            db_path=db_path,
            model_type="bert",
            template_type="test",
            template_content=test_template,
            validate=True
        )
        assert success, "Failed to store template"
        
        # Test get_template
        template, parent, modality = get_template(
            db_path=db_path,
            model_type="bert",
            template_type="test"
        )
        assert template is not None, "Failed to get template"
        assert template.strip() == test_template.strip(), "Retrieved template doesn't match stored template"
        
        # Test list_templates
        templates = list_templates(db_path)
        assert len(templates) > 0, "No templates found in database"
        
        # Test add_default_parent_templates
        success = add_default_parent_templates(db_path)
        assert success, "Failed to add default parent templates"
        
        # Test update_template_inheritance
        success = update_template_inheritance(db_path)
        assert success, "Failed to update template inheritance"
        
        # Test validate_all_templates
        results = validate_all_templates(db_path)
        assert results["total"] > 0, "No templates validated"
        
        logger.info("Template database test passed!")
    finally:
        # Clean up temporary database file
        if os.path.exists(db_path):
            os.unlink(db_path)

def main():
    """Main function to run all tests"""
    print("Testing template_utilities package...")
    
    try:
        test_placeholder_helpers()
        test_template_validation()
        test_template_inheritance()
        test_template_database()
        
        print("\nAll tests passed!")
        return 0
    except AssertionError as e:
        logger.error(f"Test failed: {e}")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error during tests: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())