#!/usr/bin/env python
# Demonstration of the template inheritance system

import os
import sys
import logging
import argparse
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import template system
try:
    from template_inheritance_system import (
        register_template, register_template_directory,
        validate_all_templates, test_template_compatibility,
        get_template_inheritance_graph, get_template_for_model,
        create_specialized_template
    )
except ImportError:
    logger.error("template_inheritance_system.py not found in current directory")
    sys.exit(1)

def demonstrate_template_system(templates_dir):
    """Demonstrate the template inheritance system"""
    logger.info("Demonstrating Template Inheritance System")
    logger.info("======================================")
    
    # Step 1: Register all templates
    logger.info("\nStep 1: Registering templates from {templates_dir}")
    templates = register_template_directory(templates_dir)
    
    logger.info(f"Registered {len(templates)} templates:")
    for template in templates:
        logger.info(f"- {template.name}")
    
    # Step 2: Validate all templates
    logger.info("\nStep 2: Validating all templates")
    validation_results = validate_all_templates()
    
    valid_count = sum(1 for r in validation_results.values() if r['valid'])
    invalid_count = len(validation_results) - valid_count
    
    logger.info(f"Validation results: {valid_count} valid, {invalid_count} invalid")
    
    if invalid_count > 0:
        logger.info("\nInvalid templates:")
        for template_name, result in validation_results.items():
            if not result['valid']:
                logger.info(f"- {template_name}: {len(result['errors'])} errors")
                for error in result['errors']:
                    logger.info(f"  - {error}")
    
    # Step 3: Show inheritance relationships
    logger.info("\nStep 3: Displaying template inheritance relationships")
    inheritance_graph = get_template_inheritance_graph()
    
    if inheritance_graph:
        for parent, children in inheritance_graph.items():
            logger.info(f"Template '{parent}' is inherited by:")
            for child in children:
                logger.info(f"  - {child}")
    else:
        logger.info("No inheritance relationships found.")
    
    # Step 4: Test template selection for different models
    logger.info("\nStep 4: Testing template selection for different models")
    test_models = [
        "bert-base-uncased",
        "t5-small",
        "gpt2",
        "roberta-base",
        "facebook/bart-base"
    ]
    
    for model_name in test_models:
        try:
            template = get_template_for_model(model_name)
            logger.info(f"For model '{model_name}': Selected template '{template.name}'")
        except Exception as e:
            logger.info(f"For model '{model_name}': Error selecting template: {str(e)}")
    
    # Step 5: Creating a specialized template
    logger.info("\nStep 5: Creating a specialized template")
    try:
        overrides = {
            "class_definition": """class TestSpecializedBertModel:
    \"\"\"Test class for specialized BERT model with custom functionality\"\"\"
    
    # Model name to load - this should be overridden
    model_name = "{{ model_name }}"
""",
            "test_custom_functionality": """def test_custom_functionality(self):
    \"\"\"Test custom functionality specific to this template\"\"\"
    logger.info("Running custom test function")
    assert self.model is not None, "Model should be loaded"
    assert self.tokenizer is not None, "Tokenizer should be loaded"
    logger.info("Custom test passed!")"""
        }
        
        specialized_template = create_specialized_template(
            "hf_bert_template.py",
            "hf_specialized_bert_template.py",
            overrides
        )
        
        logger.info(f"Created specialized template: {specialized_template.name}")
        logger.info(f"Parent templates: {specialized_template.parent_templates}")
        logger.info(f"Sections: {list(specialized_template.sections.keys())}")
    except Exception as e:
        logger.error(f"Error creating specialized template: {str(e)}")
    
    # Step 6: Generate a test file from a template
    logger.info("\nStep 6: Generating test file from template")
    try:
        bert_template = next((t for t in templates if t.name == "hf_bert_template.py"), None)
        if bert_template:
            # Render template with context
            context = {
                "model_name": "bert-base-uncased",
                "test_name": "TestGeneratedBertModel"
            }
            rendered_content = bert_template.render(context)
            
            # Save to file
            output_path = "generated_bert_test.py"
            with open(output_path, "w") as f:
                f.write(rendered_content)
            
            logger.info(f"Generated test file: {output_path}")
            
            # Show first few lines
            content_preview = "\n".join(rendered_content.split("\n")[:10])
            logger.info(f"Preview of generated file:\n{content_preview}...")
        else:
            logger.error("BERT template not found")
    except Exception as e:
        logger.error(f"Error generating test file: {str(e)}")
    
    logger.info("\nTemplate inheritance system demonstration complete!")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Template inheritance system demo")
    parser.add_argument("--templates-dir", type=str, default="./templates",
                       help="Directory containing templates")
    args = parser.parse_args()
    
    # Check if templates directory exists
    if not os.path.isdir(args.templates_dir):
        logger.error(f"Templates directory not found: {args.templates_dir}")
        sys.exit(1)
    
    # Run demonstration
    demonstrate_template_system(args.templates_dir)

if __name__ == "__main__":
    main()