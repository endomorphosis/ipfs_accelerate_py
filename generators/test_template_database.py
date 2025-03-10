#!/usr/bin/env python
"""
Test for the new DuckDB-powered hardware test template database system.

This script demonstrates the usage of the DuckDB API for hardware test templates.
It shows:
1. Listing templates and hardware platforms
2. Retrieving and displaying template details
3. Creating a new template
4. Searching for templates
"""

import sys
import os
import argparse
from pathlib import Path

# Add parent directory to path for module imports
sys.path.append(str(Path(__file__).parent))

from hardware_test_templates.template_database import TemplateDatabase

def list_all_templates(db):
    """List all templates in the database."""
    templates = db.list_templates()
    print(f"Found {len(templates)} templates:")
    for t in templates:
        print(f"  - {t['model_id']} ({t['model_family']}, {t['modality']})")
    print()

def list_hardware_platforms(db):
    """List all hardware platforms in the database."""
    hardware = db.get_hardware_platforms()
    print("Supported hardware platforms:")
    for hw in hardware:
        print(f"  - {hw['hardware_type']} ({hw['display_name']}): {hw['description']} [{hw['status']}]")
    print()

def show_template_details(db, model_id):
    """Show details for a specific template."""
    template = db.get_template_with_metadata(model_id)
    if template:
        print(f"Template for {template['model_id']}:")
        print(f"Model name: {template['model_name']}")
        print(f"Family: {template['model_family']}")
        print(f"Modality: {template['modality']}")
        print(f"Last updated: {template['last_updated']}")
        print("\nTemplate content (first 100 chars):")
        print(template['template_content'][:100] + "..." if len(template['template_content']) > 100 else template['template_content'])
    else:
        print(f"Template not found for model_id: {model_id}")
    print()

def create_new_template(db, model_id, model_name, model_family, modality):
    """Create a new template in the database."""
    # Simple template content for demonstration
    template_content = f'''"""
Hugging Face test template for {model_id} model.

This template is generated with DuckDB API for demonstration purposes.
Model Family: {model_family}
Modality: {modality}
"""

def main():
    """Run the test."""
    print(f"This is a test template for {model_id} model")
    print(f"Model name: {model_name}")
    print(f"Family: {model_family}")
    print(f"Modality: {modality}")
    
    # Add your test implementation here
    
    return True

if __name__ == "__main__":
    main()
'''
    
    # Store template in database
    success = db.store_template(
        model_id=model_id,
        template_content=template_content,
        model_name=model_name,
        model_family=model_family,
        modality=modality
    )
    
    if success:
        print(f"Created new template for {model_id}")
        # Show the template details
        show_template_details(db, model_id)
    else:
        print(f"Failed to create template for {model_id}")
    print()

def search_templates(db, query):
    """Search for templates by query string."""
    templates = db.search_templates(query)
    print(f"Found {len(templates)} templates matching '{query}':")
    for t in templates:
        print(f"  - {t['model_id']} ({t['model_family']}, {t['modality']})")
    print()

def main():
    """Main function to demonstrate the template database."""
    parser = argparse.ArgumentParser(description="Test the template database API")
    parser.add_argument("--create", action="store_true", help="Create a new template")
    parser.add_argument("--model-id", help="Model ID to show details for")
    parser.add_argument("--family", help="Filter templates by model family")
    parser.add_argument("--modality", help="Filter templates by modality")
    parser.add_argument("--search", help="Search for templates")
    parser.add_argument("--all", action="store_true", help="Show all templates")
    parser.add_argument("--hardware", action="store_true", help="Show hardware platforms")
    parser.add_argument("--db-path", default="./template_db.duckdb", help="Path to the template database")
    args = parser.parse_args()
    
    # Initialize database
    db = TemplateDatabase(db_path=args.db_path)
    
    print("=== Template Database API Demo ===\n")
    
    # Show hardware platforms
    if args.hardware or args.all:
        list_hardware_platforms(db)
    
    # List templates
    if args.all:
        list_all_templates(db)
    elif args.family or args.modality:
        templates = db.list_templates(model_family=args.family, modality=args.modality)
        filter_desc = []
        if args.family:
            filter_desc.append(f"family='{args.family}'")
        if args.modality:
            filter_desc.append(f"modality='{args.modality}'")
        print(f"Found {len(templates)} templates with {' and '.join(filter_desc)}:")
        for t in templates:
            print(f"  - {t['model_id']} ({t['model_family']}, {t['modality']})")
        print()
    
    # Show template details
    if args.model_id:
        show_template_details(db, args.model_id)
    
    # Search templates
    if args.search:
        search_templates(db, args.search)
    
    # Create new template
    if args.create:
        if not args.model_id:
            print("Error: --model-id is required when creating a template")
            return
        # Use defaults if not provided
        model_name = args.model_id.upper()
        model_family = args.family or "custom"
        modality = args.modality or "unknown"
        create_new_template(db, args.model_id, model_name, model_family, modality)
    
    print("=== Demo Complete ===")

if __name__ == "__main__":
    main()