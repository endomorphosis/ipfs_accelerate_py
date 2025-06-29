#!/usr/bin/env python3
"""
Test script for template database migration.
This script tests the migration of test generator from static templates to database templates.
"""

import os
import sys
import logging
import subprocess
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_prerequisites():
    """Check if required packages are installed"""
    try:
        import duckdb
        logger.info("DuckDB is available")
    except ImportError:
        logger.error("DuckDB is not installed. Please install it with: pip install duckdb")
        return False
    
    return True

def create_template_database():
    """Create template database for testing"""
    logger.info("Creating template database...")
    try:
        command = [sys.executable, "create_template_database.py", "--create"]
        result = subprocess.run(command, capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.error(f"Failed to create template database: {result.stderr}")
            return False
        
        logger.info("Template database created successfully")
        return True
    except Exception as e:
        logger.error(f"Error creating template database: {e}")
        return False

def test_static_templates():
    """Test generator with static templates"""
    logger.info("Testing generator with static templates...")
    try:
        command = [
            sys.executable, 
            "test_generator_with_resource_pool.py", 
            "--model", "bert-base-uncased", 
            "--output-dir", "./test_output_static"
        ]
        result = subprocess.run(command, capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.error(f"Failed to run generator with static templates: {result.stderr}")
            return False
        
        logger.info("Generator with static templates ran successfully")
        # Check if output file exists
        output_file = Path("./test_output_static/test_hf_bert_base_uncased.py")
        if not output_file.exists():
            logger.error(f"Output file {output_file} not found")
            return False
        
        logger.info(f"Output file {output_file} created successfully")
        return True
    except Exception as e:
        logger.error(f"Error testing generator with static templates: {e}")
        return False

def test_db_templates():
    """Test generator with database templates"""
    logger.info("Testing generator with database templates...")
    try:
        command = [
            sys.executable, 
            "test_generator_with_resource_pool.py", 
            "--model", "bert-base-uncased", 
            "--output-dir", "./test_output_db", 
            "--use-db-templates"
        ]
        result = subprocess.run(command, capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.error(f"Failed to run generator with database templates: {result.stderr}")
            return False
        
        logger.info("Generator with database templates ran successfully")
        # Check if output file exists
        output_file = Path("./test_output_db/test_hf_bert_base_uncased.py")
        if not output_file.exists():
            logger.error(f"Output file {output_file} not found")
            return False
        
        logger.info(f"Output file {output_file} created successfully")
        return True
    except Exception as e:
        logger.error(f"Error testing generator with database templates: {e}")
        return False

def main():
    """Main function"""
    logger.info("Testing template database migration...")
    
    # Check prerequisites
    if not check_prerequisites():
        return 1
    
    # Create directories for test output
    os.makedirs("./test_output_static", exist_ok=True)
    os.makedirs("./test_output_db", exist_ok=True)
    
    # Create template database
    if not create_template_database():
        return 1
    
    # Test generator with static templates
    if not test_static_templates():
        return 1
    
    # Test generator with database templates
    if not test_db_templates():
        return 1
    
    logger.info("All tests passed successfully!")
    logger.info("Template database migration completed.")
    return 0

if __name__ == "__main__":
    sys.exit(main())