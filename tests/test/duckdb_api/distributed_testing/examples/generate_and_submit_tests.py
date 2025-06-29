#!/usr/bin/env python3
"""
Example script demonstrating how to use the TestGeneratorIntegration class to:
1. Initialize a template database
2. Add templates for different model families
3. Add model-to-family mappings
4. Generate and submit tests for models

Usage:
    python generate_and_submit_tests.py --setup-db
    python generate_and_submit_tests.py --generate-tests
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("template_generator_demo")

# Add parent directories to path
parent_dir = str(Path(__file__).parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import TestGeneratorIntegration class
try:
    from duckdb_api.distributed_testing.test_generator_integration import TestGeneratorIntegration
except ImportError:
    logger.error("Cannot import TestGeneratorIntegration. Make sure the module is in your Python path.")
    sys.exit(1)

# Define model families and corresponding templates
MODEL_FAMILIES = {
    "text_embedding": ["bert-base-uncased", "roberta-base", "distilbert-base-uncased"],
    "vision": ["vit-base-patch16-224", "google/vit-base-patch16-224", "facebook/deit-base-patch16-224"]
}

# Hardware types to test on
HARDWARE_TYPES = ["cpu", "cuda"]

# Batch sizes to test
BATCH_SIZES = [1, 4, 16]

# Template file paths
TEMPLATE_FILES = {
    "text_embedding": os.path.join(os.path.dirname(__file__), "text_embedding_template.py"),
    "vision": os.path.join(os.path.dirname(__file__), "vision_template.py")
}

def setup_template_database(db_path):
    """Set up the template database with templates and model mappings."""
    logger.info(f"Setting up template database at {db_path}")
    
    # Create TestGeneratorIntegration instance
    generator = TestGeneratorIntegration(db_path)
    
    try:
        # Add templates for each model family
        for family, template_file in TEMPLATE_FILES.items():
            # Check if template file exists
            if not os.path.exists(template_file):
                logger.error(f"Template file not found: {template_file}")
                continue
                
            # Read template content
            with open(template_file, 'r') as f:
                template_content = f.read()
                
            # Add template to database
            template_id = generator.add_template(
                template_name=f"{family}_template",
                model_family=family,
                content=template_content,
                description=f"Template for {family} models"
            )
            
            if template_id:
                logger.info(f"Added template for {family} models with ID {template_id}")
                
                # Add model mappings for this family
                if family in MODEL_FAMILIES:
                    for model_name in MODEL_FAMILIES[family]:
                        success = generator.add_model_mapping(
                            model_name=model_name,
                            model_family=family,
                            template_id=template_id,
                            description=f"Mapping for {model_name}"
                        )
                        
                        if success:
                            logger.info(f"Added mapping for {model_name} to {family}")
                        else:
                            logger.error(f"Failed to add mapping for {model_name}")
            else:
                logger.error(f"Failed to add template for {family}")
    finally:
        # Close the database connection
        generator.close()
        
    logger.info("Template database setup complete")

def generate_tests(db_path, output_dir=None):
    """Generate tests for models using templates from the database."""
    logger.info(f"Generating tests from template database at {db_path}")
    
    # Create output directory if specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Create TestGeneratorIntegration instance
    generator = TestGeneratorIntegration(db_path)
    
    try:
        # Generate tests for each model family
        for family, models in MODEL_FAMILIES.items():
            for model_name in models:
                logger.info(f"Generating tests for {model_name}")
                
                # Generate tests
                success, tests = generator.generate_and_submit_tests(
                    model_name=model_name,
                    hardware_types=HARDWARE_TYPES,
                    batch_sizes=BATCH_SIZES
                )
                
                if success:
                    logger.info(f"Generated {len(tests)} tests for {model_name}")
                    
                    # Save generated tests to files if output_dir is specified
                    if output_dir:
                        for i, test in enumerate(tests):
                            hardware = test["hardware_type"]
                            batch_size = test["batch_size"]
                            
                            # Create filename
                            filename = f"test_{model_name.replace('/', '_').replace('-', '_')}_{hardware}_batch{batch_size}.py"
                            file_path = os.path.join(output_dir, filename)
                            
                            # Write test content to file
                            with open(file_path, "w") as f:
                                f.write(test["test_content"])
                                
                            logger.info(f"Saved test to {file_path}")
                else:
                    logger.error(f"Failed to generate tests for {model_name}")
    finally:
        # Close the database connection
        generator.close()
        
    logger.info("Test generation complete")

def main():
    parser = argparse.ArgumentParser(description="Template-Based Test Generator Demo")
    parser.add_argument("--db-path", default="./templates.duckdb", help="Path to template database")
    parser.add_argument("--setup-db", action="store_true", help="Set up the template database")
    parser.add_argument("--generate-tests", action="store_true", help="Generate tests from templates")
    parser.add_argument("--output-dir", help="Directory to save generated tests")
    
    args = parser.parse_args()
    
    if args.setup_db:
        setup_template_database(args.db_path)
        
    if args.generate_tests:
        generate_tests(args.db_path, args.output_dir)
        
    if not args.setup_db and not args.generate_tests:
        parser.print_help()

if __name__ == "__main__":
    main()