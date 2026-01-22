#\!/usr/bin/env python3
"""
Sample script to demonstrate using the enhanced generators.
"""

import os
import sys
import subprocess
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_tests():
    """Generate various tests using the improved generators."""
    
    output_dir = "./generated_tests"
    
    # 1. Generate tests for a specific model with template support
    logger.info("Generating test for BERT with template support...")
    subprocess.run([
        "python", "fixed_merged_test_generator.py",
        "--generate", "bert",
        "--use-db-templates",
        "--output-dir", output_dir
    ])
    
    # 2. Generate a test for T5 with basic generation (no template)
    logger.info("Generating test for T5 with basic generation...")
    subprocess.run([
        "python", "fixed_merged_test_generator.py",
        "--generate", "t5",
        "--output-dir", output_dir
    ])
    
    # 3. Generate skill files using templates
    logger.info("Generating skill for BERT with template support...")
    subprocess.run([
        "python", "integrated_skillset_generator.py",
        "--model", "bert",
        "--use-db-templates",
        "--output-dir", "./generated_skills"
    ])
    
    # 4. Generate tests for multiple platforms
    logger.info("Generating test for ViT on specific platforms...")
    subprocess.run([
        "python", "fixed_merged_test_generator.py",
        "--generate", "vit",
        "--platform", "cpu,cuda,openvino",
        "--output-dir", output_dir
    ])
    
    # 5. Generate tests for all key models
    logger.info("Generating tests for all key models...")
    subprocess.run([
        "python", "fixed_merged_test_generator.py",
        "--all-models",
        "--output-dir", output_dir
    ])

if __name__ == "__main__":
    generate_tests()
