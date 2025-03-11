#!/usr/bin/env python3
"""
Template Utilities Command Line Interface

This module provides a command-line interface for the template utilities package,
including commands for:
- Template validation
- Inheritance management
- Placeholder handling
- Database operations
"""

import os
import sys
import json
import logging
import argparse
import datetime
from typing import Dict, Any, List, Optional

from .placeholder_helpers import (
    get_standard_placeholders,
    get_default_context,
    render_template,
    extract_placeholders,
    detect_hardware
)
from .template_validation import (
    validate_template_syntax,
    validate_hardware_support,
    validate_template
)
from .template_inheritance import (
    get_parent_for_model_type,
    get_inheritance_hierarchy,
    get_default_parent_templates
)
from .template_database import (
    check_database,
    create_schema,
    get_db_connection,
    get_template,
    store_template,
    list_templates,
    add_default_parent_templates,
    update_template_inheritance,
    validate_all_templates
)

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Default database path
DEFAULT_DB_PATH = "./template_db.duckdb"

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Template utilities command line interface"
    )
    
    # Global options
    parser.add_argument(
        "--db-path", type=str, default=DEFAULT_DB_PATH,
        help=f"Path to template database file (default: {DEFAULT_DB_PATH})"
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Enable debug logging"
    )
    
    # Subparsers for commands
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Database commands
    db_parser = subparsers.add_parser("db", help="Database operations")
    db_parser.add_argument(
        "--create", action="store_true",
        help="Create database and tables if they don't exist"
    )
    db_parser.add_argument(
        "--check", action="store_true",
        help="Check if database exists and has proper schema"
    )
    db_parser.add_argument(
        "--list", action="store_true",
        help="List all templates in the database"
    )
    db_parser.add_argument(
        "--model-type", type=str,
        help="Filter templates by model type"
    )
    db_parser.add_argument(
        "--add-defaults", action="store_true",
        help="Add default parent templates to the database"
    )
    
    # Template commands
    template_parser = subparsers.add_parser("template", help="Template operations")
    template_parser.add_argument(
        "--get", action="store_true",
        help="Get a template from the database"
    )
    template_parser.add_argument(
        "--model-type", type=str, required=True,
        help="Model type to get template for"
    )
    template_parser.add_argument(
        "--template-type", type=str, required=True,
        help="Template type to get"
    )
    template_parser.add_argument(
        "--hardware-platform", type=str,
        help="Hardware platform to get template for"
    )
    template_parser.add_argument(
        "--output", type=str,
        help="Output file to write template to (if not specified, prints to stdout)"
    )
    template_parser.add_argument(
        "--store", action="store_true",
        help="Store a template in the database"
    )
    template_parser.add_argument(
        "--input", type=str,
        help="Input file containing template content to store"
    )
    template_parser.add_argument(
        "--parent", type=str,
        help="Parent template name"
    )
    template_parser.add_argument(
        "--modality", type=str,
        help="Template modality"
    )
    template_parser.add_argument(
        "--no-validate", action="store_true",
        help="Skip validation when storing template"
    )
    
    # Validation commands
    validation_parser = subparsers.add_parser("validate", help="Template validation")
    validation_parser.add_argument(
        "--template", type=str,
        help="Path to template file to validate"
    )
    validation_parser.add_argument(
        "--model-type", type=str, required=True,
        help="Model type the template is for"
    )
    validation_parser.add_argument(
        "--template-type", type=str, required=True,
        help="Template type (test, benchmark, skill)"
    )
    validation_parser.add_argument(
        "--hardware-platform", type=str,
        help="Hardware platform to validate for"
    )
    validation_parser.add_argument(
        "--all", action="store_true",
        help="Validate all templates in the database"
    )
    validation_parser.add_argument(
        "--output", type=str,
        help="Output file to write validation results to (if not specified, prints to stdout)"
    )
    
    # Inheritance commands
    inheritance_parser = subparsers.add_parser("inheritance", help="Template inheritance")
    inheritance_parser.add_argument(
        "--update", action="store_true",
        help="Update all templates in the database with inheritance information"
    )
    inheritance_parser.add_argument(
        "--get-parent", action="store_true",
        help="Get parent template for a model type"
    )
    inheritance_parser.add_argument(
        "--model-type", type=str,
        help="Model type to get parent for"
    )
    inheritance_parser.add_argument(
        "--get-hierarchy", action="store_true",
        help="Get inheritance hierarchy for a model type"
    )
    
    # Placeholder commands
    placeholder_parser = subparsers.add_parser("placeholder", help="Placeholder operations")
    placeholder_parser.add_argument(
        "--list-standard", action="store_true",
        help="List standard placeholders and their properties"
    )
    placeholder_parser.add_argument(
        "--extract", action="store_true",
        help="Extract placeholders from a template"
    )
    placeholder_parser.add_argument(
        "--template", type=str,
        help="Path to template file to extract placeholders from"
    )
    placeholder_parser.add_argument(
        "--context", action="store_true",
        help="Generate default context for a model"
    )
    placeholder_parser.add_argument(
        "--model-name", type=str,
        help="Model name to generate context for"
    )
    placeholder_parser.add_argument(
        "--hardware-platform", type=str,
        help="Hardware platform to use for context generation"
    )
    placeholder_parser.add_argument(
        "--output", type=str,
        help="Output file to write context to (if not specified, prints to stdout)"
    )
    placeholder_parser.add_argument(
        "--render", action="store_true",
        help="Render a template with context"
    )
    placeholder_parser.add_argument(
        "--context-file", type=str,
        help="JSON file containing context for rendering"
    )
    
    # Hardware commands
    hardware_parser = subparsers.add_parser("hardware", help="Hardware operations")
    hardware_parser.add_argument(
        "--detect", action="store_true",
        help="Detect available hardware platforms"
    )
    hardware_parser.add_argument(
        "--validate", action="store_true",
        help="Validate hardware support in a template"
    )
    hardware_parser.add_argument(
        "--template", type=str,
        help="Path to template file to validate hardware support for"
    )
    hardware_parser.add_argument(
        "--hardware-platform", type=str,
        help="Hardware platform to validate for"
    )
    
    return parser.parse_args()

def setup_environment(args):
    """Set up the environment for command execution"""
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")

def execute_db_command(args):
    """Execute database command"""
    if args.create:
        # Create database and tables
        import duckdb
        
        try:
            # Check if database exists
            if os.path.exists(args.db_path):
                logger.info(f"Database file {args.db_path} already exists")
                
                # Check if database has the right schema
                if check_database(args.db_path):
                    logger.info("Database schema is valid")
                    return 0
                
                logger.warning("Database exists but schema is invalid, creating new schema")
            
            # Create connection and schema
            conn = get_db_connection(args.db_path)
            create_schema(conn)
            conn.close()
            
            logger.info(f"Database created at {args.db_path}")
            return 0
        except Exception as e:
            logger.error(f"Error creating database: {e}")
            return 1
    
    elif args.check:
        # Check if database exists and has proper schema
        if check_database(args.db_path):
            logger.info("Database exists and has proper schema")
            return 0
        else:
            logger.error("Database check failed")
            return 1
    
    elif args.list:
        # List all templates in the database
        templates = list_templates(args.db_path, args.model_type)
        
        if not templates:
            logger.info("No templates found in database")
            return 0
        
        # Print templates in a formatted table
        print("\nTemplates:")
        print("-" * 100)
        print(f"{'ID':<5} {'Model Type':<15} {'Template Type':<15} {'Hardware':<10} {'Status':<10} {'Modality':<12} {'Parent':<15} {'Validation':<10}")
        print("-" * 100)
        
        for template in templates:
            print(f"{template['id']:<5} {template['model_type']:<15} {template['template_type']:<15} {template['hardware_platform']:<10} {template['status']:<10} {template['modality']:<12} {template.get('parent_template', ''):<15} {template['validation_status']:<10}")
        
        print(f"\nTotal: {len(templates)} templates")
        return 0
    
    elif args.add_defaults:
        # Add default parent templates to the database
        if add_default_parent_templates(args.db_path):
            logger.info("Default parent templates added successfully")
            return 0
        else:
            logger.error("Failed to add default parent templates")
            return 1
    
    else:
        logger.error("No database command specified")
        return 1

def execute_template_command(args):
    """Execute template command"""
    if args.get:
        # Get a template from the database
        template, parent, modality = get_template(
            args.db_path,
            args.model_type,
            args.template_type,
            args.hardware_platform
        )
        
        if template:
            if args.output:
                # Write template to file
                with open(args.output, 'w') as f:
                    f.write(template)
                logger.info(f"Template written to {args.output}")
            else:
                # Print template to stdout
                print("\nTemplate:")
                print("=" * 80)
                print(template)
                print("=" * 80)
                print(f"Parent template: {parent or 'None'}")
                print(f"Modality: {modality or 'unknown'}")
            
            return 0
        else:
            logger.error(f"Template not found for {args.model_type}/{args.template_type}/{args.hardware_platform or 'generic'}")
            return 1
    
    elif args.store:
        # Store a template in the database
        template_content = None
        
        if args.input:
            # Read template from file
            with open(args.input, 'r') as f:
                template_content = f.read()
        else:
            # Read template from stdin
            logger.info("Enter template content (Ctrl+D to finish):")
            template_content = sys.stdin.read()
        
        if not template_content:
            logger.error("No template content provided")
            return 1
        
        # Store template
        if store_template(
            args.db_path,
            args.model_type,
            args.template_type,
            template_content,
            args.hardware_platform,
            args.parent,
            args.modality,
            not args.no_validate
        ):
            logger.info(f"Template stored successfully for {args.model_type}/{args.template_type}/{args.hardware_platform or 'generic'}")
            return 0
        else:
            logger.error("Failed to store template")
            return 1
    
    else:
        logger.error("No template command specified")
        return 1

def execute_validation_command(args):
    """Execute validation command"""
    if args.all:
        # Validate all templates in the database
        results = validate_all_templates(args.db_path, args.model_type)
        
        print("\nValidation Results:")
        print("-" * 50)
        print(f"Valid templates: {results['valid']}")
        print(f"Invalid templates: {results['invalid']}")
        print(f"Total templates: {results['total']}")
        
        if results["valid"] == results["total"]:
            logger.info("All templates are valid")
            return 0
        else:
            logger.warning(f"{results['invalid']} invalid templates found")
            return 1
    
    elif args.template:
        # Validate a specific template
        with open(args.template, 'r') as f:
            template_content = f.read()
        
        # Validate template
        success, validation_results = validate_template(
            template_content,
            args.template_type,
            args.model_type,
            args.hardware_platform
        )
        
        # Format validation results
        result = {
            "success": success,
            "syntax": {
                "success": validation_results["syntax"]["success"],
                "errors": validation_results["syntax"]["errors"]
            },
            "hardware": {
                "success": validation_results["hardware"]["success"],
                "support": validation_results["hardware"]["support"]
            },
            "placeholders": {
                "success": validation_results["placeholders"]["success"],
                "missing": validation_results["placeholders"]["missing"],
                "all": validation_results["placeholders"]["all"]
            }
        }
        
        # Output validation results
        if args.output:
            # Write validation results to file
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2)
            logger.info(f"Validation results written to {args.output}")
        else:
            # Print validation results to stdout
            print("\nValidation Results:")
            print("-" * 50)
            print(f"Overall: {'VALID' if success else 'INVALID'}")
            print("\nSyntax:")
            print(f"  Status: {'VALID' if result['syntax']['success'] else 'INVALID'}")
            if result["syntax"]["errors"]:
                print("  Errors:")
                for error in result["syntax"]["errors"]:
                    print(f"    - {error}")
            
            print("\nHardware Support:")
            print(f"  Status: {'SUPPORTED' if result['hardware']['success'] else 'NOT SUPPORTED'}")
            print("  Supported Hardware:")
            for platform, supported in result["hardware"]["support"].items():
                status = "✅" if supported else "❌"
                print(f"    - {platform}: {status}")
            
            print("\nPlaceholders:")
            print(f"  Status: {'VALID' if result['placeholders']['success'] else 'INVALID'}")
            if result["placeholders"]["missing"]:
                print("  Missing Required Placeholders:")
                for placeholder in result["placeholders"]["missing"]:
                    print(f"    - {placeholder}")
            
            print("  All Placeholders:")
            for placeholder in result["placeholders"]["all"]:
                print(f"    - {placeholder}")
        
        return 0 if success else 1
    
    else:
        logger.error("No validation command specified")
        return 1

def execute_inheritance_command(args):
    """Execute inheritance command"""
    if args.update:
        # Update all templates in the database with inheritance information
        if update_template_inheritance(args.db_path):
            logger.info("Template inheritance updated successfully")
            return 0
        else:
            logger.error("Failed to update template inheritance")
            return 1
    
    elif args.get_parent:
        # Get parent template for a model type
        if not args.model_type:
            logger.error("No model type specified")
            return 1
        
        parent, modality = get_parent_for_model_type(args.model_type)
        
        print(f"\nParent Template for {args.model_type}:")
        print("-" * 50)
        print(f"Parent: {parent or 'None'}")
        print(f"Modality: {modality}")
        
        return 0
    
    elif args.get_hierarchy:
        # Get inheritance hierarchy for a model type
        if not args.model_type:
            logger.error("No model type specified")
            return 1
        
        hierarchy = get_inheritance_hierarchy(args.model_type)
        
        print(f"\nInheritance Hierarchy for {args.model_type}:")
        print("-" * 50)
        for i, model_type in enumerate(hierarchy):
            prefix = "  " * i
            print(f"{prefix}{'├── ' if i > 0 else ''}{model_type}")
        
        return 0
    
    else:
        logger.error("No inheritance command specified")
        return 1

def execute_placeholder_command(args):
    """Execute placeholder command"""
    if args.list_standard:
        # List standard placeholders
        placeholders = get_standard_placeholders()
        
        print("\nStandard Placeholders:")
        print("-" * 100)
        print(f"{'Placeholder':<20} {'Required':<10} {'Default Value':<20} {'Description'}")
        print("-" * 100)
        
        for name, info in placeholders.items():
            required = "✅" if info.get("required", False) else ""
            default = info.get("default_value", "")
            if default is None:
                default = "None"
            description = info.get("description", "")
            
            print(f"{name:<20} {required:<10} {str(default):<20} {description}")
        
        return 0
    
    elif args.extract:
        # Extract placeholders from a template
        if not args.template:
            logger.error("No template file specified")
            return 1
        
        with open(args.template, 'r') as f:
            template_content = f.read()
        
        placeholders = extract_placeholders(template_content)
        
        print(f"\nPlaceholders in {args.template}:")
        print("-" * 50)
        for placeholder in sorted(placeholders):
            print(f"- {placeholder}")
        
        print(f"\nTotal: {len(placeholders)} placeholders")
        return 0
    
    elif args.context:
        # Generate default context for a model
        if not args.model_name:
            logger.error("No model name specified")
            return 1
        
        context = get_default_context(args.model_name, args.hardware_platform)
        
        if args.output:
            # Write context to file
            with open(args.output, 'w') as f:
                json.dump(context, f, indent=2)
            logger.info(f"Context written to {args.output}")
        else:
            # Print context to stdout
            print("\nDefault Context:")
            print("-" * 50)
            for key, value in context.items():
                print(f"{key}: {value}")
        
        return 0
    
    elif args.render:
        # Render a template with context
        if not args.template:
            logger.error("No template file specified")
            return 1
        
        with open(args.template, 'r') as f:
            template_content = f.read()
        
        context = {}
        
        if args.context_file:
            # Read context from file
            with open(args.context_file, 'r') as f:
                context = json.load(f)
        elif args.model_name:
            # Generate default context
            context = get_default_context(args.model_name, args.hardware_platform)
        else:
            logger.error("No context file or model name specified")
            return 1
        
        # Render template
        rendered = render_template(template_content, context)
        
        if args.output:
            # Write rendered template to file
            with open(args.output, 'w') as f:
                f.write(rendered)
            logger.info(f"Rendered template written to {args.output}")
        else:
            # Print rendered template to stdout
            print("\nRendered Template:")
            print("=" * 80)
            print(rendered)
            print("=" * 80)
        
        return 0
    
    else:
        logger.error("No placeholder command specified")
        return 1

def execute_hardware_command(args):
    """Execute hardware command"""
    if args.detect:
        # Detect available hardware platforms
        hardware = detect_hardware()
        
        print("\nDetected Hardware:")
        print("-" * 50)
        for platform, available in hardware.items():
            status = "✅ Available" if available else "❌ Not Available"
            print(f"{platform:<10}: {status}")
        
        return 0
    
    elif args.validate:
        # Validate hardware support in a template
        if not args.template:
            logger.error("No template file specified")
            return 1
        
        with open(args.template, 'r') as f:
            template_content = f.read()
        
        success, hardware_support = validate_hardware_support(
            template_content,
            args.hardware_platform
        )
        
        print("\nHardware Support Validation:")
        print("-" * 50)
        print(f"Overall: {'SUPPORTED' if success else 'NOT SUPPORTED'}")
        print("\nSupported Hardware:")
        for platform, supported in hardware_support.items():
            status = "✅" if supported else "❌"
            print(f"{platform:<10}: {status}")
        
        return 0 if success else 1
    
    else:
        logger.error("No hardware command specified")
        return 1

def main():
    """Main entry point"""
    args = parse_args()
    setup_environment(args)
    
    # Execute command based on subparser
    if args.command == "db":
        return execute_db_command(args)
    elif args.command == "template":
        return execute_template_command(args)
    elif args.command == "validate":
        return execute_validation_command(args)
    elif args.command == "inheritance":
        return execute_inheritance_command(args)
    elif args.command == "placeholder":
        return execute_placeholder_command(args)
    elif args.command == "hardware":
        return execute_hardware_command(args)
    else:
        logger.error("No command specified")
        return 1

if __name__ == "__main__":
    sys.exit(main())