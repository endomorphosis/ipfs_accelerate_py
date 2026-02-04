#!/usr/bin/env python3
"""
Template database migration validator.
Validates the database-driven template system.
"""

import os
import sys
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_database_exists():
    """Check if template database exists"""
    try:
        # Import function from create_template_database
        from create_template_database import DEFAULT_DB_PATH
        
        # Check if database file exists
        db_path = Path(DEFAULT_DB_PATH)
        if not db_path.exists():
            logger.error(f"Template database file {DEFAULT_DB_PATH} does not exist")
            return False
        
        logger.info(f"Template database file {DEFAULT_DB_PATH} exists")
        return True
    except ImportError as e:
        logger.error(f"Failed to import from create_template_database: {e}")
        return False

def list_templates():
    """List templates in the database"""
    try:
        # Import function from create_template_database
        from create_template_database import list_templates as db_list_templates, DEFAULT_DB_PATH
        
        # Call the list_templates function
        if not db_list_templates(DEFAULT_DB_PATH):
            logger.error(f"Failed to list templates in {DEFAULT_DB_PATH}")
            return False
        
        return True
    except ImportError as e:
        logger.error(f"Failed to import from create_template_database: {e}")
        return False

def get_and_display_template():
    """Get a template from the database and display it"""
    try:
        # Import function from create_template_database
        from create_template_database import get_template_from_db, DEFAULT_DB_PATH
        
        # Get a template from the database
        template = get_template_from_db(DEFAULT_DB_PATH, "bert", "test")
        
        if template:
            logger.info("Successfully retrieved template for bert/test")
            print("\nTemplate preview (first 200 characters):")
            print("-" * 80)
            print(template[:200] + "...")
            print("-" * 80)
            return True
        else:
            logger.error("Failed to retrieve template for bert/test")
            return False
    except ImportError as e:
        logger.error(f"Failed to import from create_template_database: {e}")
        return False

def main():
    """Main function"""
    logger.info("Validating template database...")
    
    # Check if database exists
    if not check_database_exists():
        logger.error("Database check failed. Make sure to run create_template_database.py first")
        return 1
    
    # List templates in database
    if not list_templates():
        logger.error("Failed to list templates in database")
        return 1
    
    # Get and display a template
    if not get_and_display_template():
        logger.error("Failed to get and display template")
        return 1
    
    logger.info("Template database validation completed successfully")
    return 0

if __name__ == "__main__":
    sys.exit(main())