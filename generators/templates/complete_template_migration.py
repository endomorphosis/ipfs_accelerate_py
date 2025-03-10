#!/usr/bin/env python3
"""
Complete template migration script.

This script automates the entire process of completing the template migration from
static files to the DuckDB-based template system. It performs the following steps:

1. Create or update the template database schema
2. Import existing template files to the database
3. Fix syntax errors in problematic templates
4. Add missing hardware platform support
5. Update generators to use database templates
6. Validate the final system
"""

import os
import sys
import json
import logging
import argparse
import subprocess
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from templates.template_database import TemplateDatabase

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_cmd(cmd: str) -> Tuple[int, str, str]:
    """Run a command and return the result.
    
    Args:
        cmd: Command to run
        
    Returns:
        Tuple of (return_code, stdout, stderr)
    """
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=True,
        universal_newlines=True
    )
    stdout, stderr = process.communicate()
    return process.returncode, stdout, stderr

def setup_template_database(db_path: str, json_path: str) -> bool:
    """Create or update the template database.
    
    Args:
        db_path: Path to the DuckDB database
        json_path: Path to the JSON fallback file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        logger.info("Setting up template database")
        
        # Create the database
        db = TemplateDatabase(db_path, json_path)
        
        # Check if database has been created
        if db.conn is None:
            logger.error("Failed to create database connection")
            return False
        
        # Export JSON fallback for backup
        json_backup = f"{json_path}.bak_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        if os.path.exists(json_path):
            with open(json_path, 'r') as f_in:
                with open(json_backup, 'w') as f_out:
                    f_out.write(f_in.read())
            logger.info(f"Created backup of JSON templates at {json_backup}")
        
        logger.info("Template database setup complete")
        return True
    except Exception as e:
        logger.error(f"Error setting up template database: {e}")
        return False

def import_template_files(db_path: str, json_path: str, template_dir: str) -> bool:
    """Import template files to the database.
    
    Args:
        db_path: Path to the DuckDB database
        json_path: Path to the JSON fallback file
        template_dir: Directory containing template files
        
    Returns:
        True if successful, False otherwise
    """
    try:
        logger.info(f"Importing templates from {template_dir}")
        
        # Initialize database
        db = TemplateDatabase(db_path, json_path)
        
        # Check if database directory has templates
        if not os.path.isdir(template_dir):
            logger.error(f"Template directory not found: {template_dir}")
            return False
        
        # Import templates from directory
        success = db.import_from_directory(template_dir)
        
        if success:
            logger.info("Templates imported successfully")
            # Export to JSON as backup
            db.export_to_json(json_path)
            logger.info(f"Templates exported to {json_path}")
            return True
        else:
            logger.error("Failed to import templates")
            return False
    except Exception as e:
        logger.error(f"Error importing templates: {e}")
        return False

def fix_template_syntax(db_path: str, json_path: str) -> bool:
    """Fix syntax errors in templates.
    
    Args:
        db_path: Path to the DuckDB database
        json_path: Path to the JSON fallback file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        logger.info("Fixing template syntax errors")
        
        # Run the fix_templates.py script
        cmd = f"python {os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fix_templates.py')} --fix-all --db-path {db_path} --json-path {json_path}"
        return_code, stdout, stderr = run_cmd(cmd)
        
        if return_code != 0:
            logger.error(f"Failed to fix template syntax: {stderr}")
            return False
        
        logger.info(stdout)
        logger.info("Template syntax errors fixed")
        return True
    except Exception as e:
        logger.error(f"Error fixing template syntax: {e}")
        return False

def add_hardware_support(db_path: str, json_path: str) -> bool:
    """Add missing hardware platform support to templates.
    
    Args:
        db_path: Path to the DuckDB database
        json_path: Path to the JSON fallback file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        logger.info("Adding missing hardware platform support")
        
        # Run the fix_templates.py script with hardware support flag
        cmd = f"python {os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fix_templates.py')} --add-hardware-support --db-path {db_path} --json-path {json_path}"
        return_code, stdout, stderr = run_cmd(cmd)
        
        if return_code != 0:
            logger.error(f"Failed to add hardware support: {stderr}")
            return False
        
        logger.info(stdout)
        logger.info("Hardware platform support added")
        return True
    except Exception as e:
        logger.error(f"Error adding hardware support: {e}")
        return False

def update_generators(base_dir: str) -> bool:
    """Update generators to use database templates.
    
    Args:
        base_dir: Base directory containing generators
        
    Returns:
        True if successful, False otherwise
    """
    try:
        logger.info("Updating generators to use database templates")
        
        # Run the update_generators_to_use_db.py script
        cmd = f"python {os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'test_generators', 'update_generators_to_use_db.py')} --dir {base_dir}"
        return_code, stdout, stderr = run_cmd(cmd)
        
        if return_code != 0:
            logger.error(f"Failed to update generators: {stderr}")
            return False
        
        logger.info(stdout)
        logger.info("Generators updated to use database templates")
        return True
    except Exception as e:
        logger.error(f"Error updating generators: {e}")
        return False

def validate_system(db_path: str, json_path: str, base_dir: str, model_type: str = "bert") -> bool:
    """Validate the entire template system.
    
    Args:
        db_path: Path to the DuckDB database
        json_path: Path to the JSON fallback file
        base_dir: Base directory containing generators
        model_type: Model type to validate with
        
    Returns:
        True if successful, False otherwise
    """
    try:
        logger.info("Validating template system")
        
        # 1. Validate templates in the database
        logger.info("Validating database templates")
        db = TemplateDatabase(db_path, json_path)
        validation_results = db.validate_all_templates()
        
        if validation_results['invalid'] > 0:
            logger.error(f"Found {validation_results['invalid']} invalid templates")
            logger.error("Template validation failed")
            return False
        
        logger.info(f"All {validation_results['valid']} templates validated successfully")
        
        # 2. Test generating a test file using a generator
        logger.info(f"Testing test file generation for {model_type}")
        test_generator_path = os.path.join(base_dir, "test_generators", "simple_test_generator.py")
        
        if not os.path.exists(test_generator_path):
            logger.warning(f"Test generator not found at {test_generator_path}")
            test_generator_path = os.path.join(base_dir, "test_generators", "merged_test_generator.py")
            
            if not os.path.exists(test_generator_path):
                logger.error("Cannot find a test generator to validate with")
                return False
        
        # Generate a test file using database templates
        output_file = os.path.join("/tmp", f"test_{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.py")
        cmd = f"python {test_generator_path} --model {model_type} --use-db-templates --output {output_file}"
        return_code, stdout, stderr = run_cmd(cmd)
        
        if return_code != 0:
            logger.error(f"Failed to generate test file: {stderr}")
            return False
        
        logger.info(stdout)
        
        # Check if output file exists and is valid
        if not os.path.exists(output_file):
            logger.error(f"Output file not created: {output_file}")
            return False
        
        # Validate the output file
        with open(output_file, 'r') as f:
            output_content = f.read()
        
        if not output_content or "ERROR" in output_content:
            logger.error(f"Generated test file contains errors:\n{output_content}")
            return False
        
        logger.info(f"Generated test file successfully: {output_file}")
        logger.info("Template system validation successful")
        return True
    except Exception as e:
        logger.error(f"Error validating system: {e}")
        return False

def update_next_steps_md(next_steps_path: str) -> bool:
    """Update the NEXT_STEPS.md file to reflect completion.
    
    Args:
        next_steps_path: Path to NEXT_STEPS.md file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        logger.info(f"Updating {next_steps_path}")
        
        if not os.path.exists(next_steps_path):
            logger.warning(f"NEXT_STEPS.md not found at {next_steps_path}")
            return False
        
        with open(next_steps_path, 'r') as f:
            content = f.read()
        
        # Update the implementation plan section
        updated_content = content.replace(
            "## Implementation Plan",
            "## Implementation Plan (COMPLETED - March 10, 2025)"
        )
        
        # Update the required next steps section
        updated_content = updated_content.replace(
            "## Required Next Steps",
            "## Required Next Steps (COMPLETED - March 10, 2025)"
        )
        
        # Add a completion note at the top
        completion_note = """
## Migration Status: COMPLETED (March 10, 2025)

The template migration from static files to DuckDB-based templates has been completed:

- ✅ Fixed all templates with syntax errors
- ✅ Added comprehensive hardware platform support to all templates
- ✅ Implemented DuckDB database integration
- ✅ Updated generators to use database templates with fallback
- ✅ Validated the template system with end-to-end testing

The `--use-db-templates` flag is now available in all generators, and will become the default in a future release.
Setting the environment variable `USE_DB_TEMPLATES=1` will also enable the database templates.

"""
        updated_content = updated_content.replace(
            "# Next Steps for Template System Improvement",
            "# Template System Migration (COMPLETED)\n\n" + completion_note
        )
        
        # Write updated content
        with open(next_steps_path, 'w') as f:
            f.write(updated_content)
        
        logger.info(f"Updated {next_steps_path}")
        return True
    except Exception as e:
        logger.error(f"Error updating NEXT_STEPS.md: {e}")
        return False

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Complete template migration from static files to database"
    )
    parser.add_argument(
        "--db-path", type=str,
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "template_db.duckdb"),
        help="Path to the DuckDB database"
    )
    parser.add_argument(
        "--json-path", type=str,
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "template_db.json"),
        help="Path to the JSON fallback file"
    )
    parser.add_argument(
        "--template-dir", type=str,
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_templates"),
        help="Directory containing template files"
    )
    parser.add_argument(
        "--base-dir", type=str,
        default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        help="Base directory containing generators"
    )
    parser.add_argument(
        "--next-steps-path", type=str,
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "NEXT_STEPS.md"),
        help="Path to NEXT_STEPS.md file"
    )
    parser.add_argument(
        "--skip-setup", action="store_true",
        help="Skip database setup and template import"
    )
    parser.add_argument(
        "--skip-fixes", action="store_true",
        help="Skip template fixes and hardware support"
    )
    parser.add_argument(
        "--skip-update", action="store_true",
        help="Skip generator updates"
    )
    parser.add_argument(
        "--skip-validate", action="store_true",
        help="Skip system validation"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Enable verbose logging"
    )
    
    return parser.parse_args()

def main():
    """Main function."""
    args = parse_args()
    
    # Configure logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")
    
    logger.info("Starting template migration completion")
    
    # 1. Setup database
    if not args.skip_setup:
        if not setup_template_database(args.db_path, args.json_path):
            logger.error("Database setup failed")
            return 1
        
        # Import templates
        if not import_template_files(args.db_path, args.json_path, args.template_dir):
            logger.error("Template import failed")
            return 1
    
    # 2. Fix template issues
    if not args.skip_fixes:
        if not fix_template_syntax(args.db_path, args.json_path):
            logger.error("Template syntax fix failed")
            return 1
        
        if not add_hardware_support(args.db_path, args.json_path):
            logger.error("Hardware support addition failed")
            return 1
    
    # 3. Update generators
    if not args.skip_update:
        if not update_generators(args.base_dir):
            logger.error("Generator update failed")
            return 1
    
    # 4. Validate system
    if not args.skip_validate:
        if not validate_system(args.db_path, args.json_path, args.base_dir):
            logger.error("System validation failed")
            return 1
    
    # 5. Update NEXT_STEPS.md
    if not update_next_steps_md(args.next_steps_path):
        logger.warning("Failed to update NEXT_STEPS.md")
    
    logger.info("Template migration completion finished successfully")
    
    # Report success
    print("\n" + "=" * 80)
    print("Template Migration Completed Successfully!")
    print("=" * 80)
    print(f"\nDatabase: {args.db_path}")
    print(f"JSON Backup: {args.json_path}")
    print("\nThe following actions were completed:")
    if not args.skip_setup:
        print("- Template database setup and import")
    if not args.skip_fixes:
        print("- Template syntax fixes and hardware support addition")
    if not args.skip_update:
        print("- Generator updates to use database templates")
    if not args.skip_validate:
        print("- System validation")
    print("\nGenerators now support the --use-db-templates flag to use database templates.")
    print("You can also set the USE_DB_TEMPLATES=1 environment variable.")
    print("=" * 80)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())