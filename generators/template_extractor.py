#!/usr/bin/env python3
"""
Template Extractor Tool

This script extracts templates from the template database for inspection and fixing.

Usage:
    python template_extractor.py --extract-template model_type/template_type [--db-path DB_PATH]
    python template_extractor.py --list-templates [--db-path DB_PATH]
    python template_extractor.py --save-template template_id fixed_content.py [--db-path DB_PATH]
"""

import os
import sys
import argparse
import json
import logging
import re
from typing import Dict, List, Any, Optional
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Check for DuckDB availability
try:
    import duckdb
    HAS_DUCKDB = True
except ImportError:
    HAS_DUCKDB = False
    logger.warning("DuckDB not available. Will use JSON-based storage.")

def list_templates(db_path: str):
    """
    List all templates in the database
    
    Args:
        db_path: Path to the database file
    """
    if not HAS_DUCKDB or db_path.endswith('.json'):
        # Use JSON-based storage
        json_db_path = db_path if db_path.endswith('.json') else db_path.replace('.duckdb', '.json')
        
        try:
            if not os.path.exists(json_db_path):
                logger.error(f"JSON database file not found: {json_db_path}")
                return
            
            # Load the JSON database
            with open(json_db_path, 'r') as f:
                template_db = json.load(f)
            
            if 'templates' not in template_db:
                logger.error("No templates found in JSON database")
                return
            
            templates = template_db['templates']
            
            print(f"Found {len(templates)} templates in JSON database:")
            for template_id, template_data in templates.items():
                model_type = template_data.get('model_type', 'unknown')
                template_type = template_data.get('template_type', 'unknown')
                platform = template_data.get('platform', 'generic')
                
                key = f"{model_type}/{template_type}"
                if platform and platform != 'generic':
                    key += f"/{platform}"
                
                updated_at = template_data.get('updated_at', 'unknown')
                
                print(f"- {template_id}: {key} (Updated: {updated_at})")
        
        except Exception as e:
            logger.error(f"Error listing templates from JSON database: {str(e)}")
    
    else:
        # Use DuckDB
        try:
            import duckdb
            
            if not os.path.exists(db_path):
                logger.error(f"Database file not found: {db_path}")
                return
            
            # Connect to the database
            conn = duckdb.connect(db_path)
            
            # Check if templates table exists
            table_check = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='templates'").fetchall()
            if not table_check:
                logger.error("No 'templates' table found in database")
                return
            
            # Get all templates
            templates = conn.execute("SELECT id, model_type, template_type, platform, updated_at FROM templates").fetchall()
            
            print(f"Found {len(templates)} templates in DuckDB database:")
            for template_id, model_type, template_type, platform, updated_at in templates:
                key = f"{model_type}/{template_type}"
                if platform and platform != 'generic':
                    key += f"/{platform}"
                
                print(f"- {template_id}: {key} (Updated: {updated_at})")
            
            conn.close()
        
        except Exception as e:
            logger.error(f"Error listing templates from DuckDB database: {str(e)}")

def extract_template(db_path: str, template_key: str) -> Optional[str]:
    """
    Extract a template from the database
    
    Args:
        db_path: Path to the database file
        template_key: Template key in format "model_type/template_type[/platform]"
        
    Returns:
        Template content as string, or None if not found
    """
    # Parse template key
    parts = template_key.split('/')
    if len(parts) < 2:
        logger.error(f"Invalid template key format: {template_key}")
        logger.error("Expected format: model_type/template_type[/platform]")
        return None
    
    model_type = parts[0]
    template_type = parts[1]
    platform = parts[2] if len(parts) > 2 else None
    
    if not HAS_DUCKDB or db_path.endswith('.json'):
        # Use JSON-based storage
        json_db_path = db_path if db_path.endswith('.json') else db_path.replace('.duckdb', '.json')
        
        try:
            if not os.path.exists(json_db_path):
                logger.error(f"JSON database file not found: {json_db_path}")
                return None
            
            # Load the JSON database
            with open(json_db_path, 'r') as f:
                template_db = json.load(f)
            
            if 'templates' not in template_db:
                logger.error("No templates found in JSON database")
                return None
            
            templates = template_db['templates']
            
            # Find matching template
            for template_id, template_data in templates.items():
                if (template_data.get('model_type') == model_type and 
                    template_data.get('template_type') == template_type):
                    
                    # Check platform if specified
                    if platform is not None:
                        if template_data.get('platform') != platform:
                            continue
                    
                    logger.info(f"Found template: {template_id} ({model_type}/{template_type})")
                    return template_data.get('template')
            
            logger.error(f"Template not found: {template_key}")
            return None
        
        except Exception as e:
            logger.error(f"Error extracting template from JSON database: {str(e)}")
            return None
    
    else:
        # Use DuckDB
        try:
            import duckdb
            
            if not os.path.exists(db_path):
                logger.error(f"Database file not found: {db_path}")
                return None
            
            # Connect to the database
            conn = duckdb.connect(db_path)
            
            # Check if templates table exists
            table_check = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='templates'").fetchall()
            if not table_check:
                logger.error("No 'templates' table found in database")
                return None
            
            # Build query
            query = "SELECT id, template FROM templates WHERE model_type = ? AND template_type = ?"
            params = [model_type, template_type]
            
            if platform is not None:
                query += " AND platform = ?"
                params.append(platform)
            
            # Get matching template
            result = conn.execute(query, params).fetchone()
            
            if result:
                template_id, template_content = result
                logger.info(f"Found template: {template_id} ({model_type}/{template_type})")
                return template_content
            
            logger.error(f"Template not found: {template_key}")
            return None
        
        except Exception as e:
            logger.error(f"Error extracting template from DuckDB database: {str(e)}")
            return None

def save_template(db_path: str, template_id: str, content_path: str) -> bool:
    """
    Save a fixed template back to the database
    
    Args:
        db_path: Path to the database file
        template_id: Template ID
        content_path: Path to the fixed template content
        
    Returns:
        True if successful, False otherwise
    """
    from datetime import datetime
    # Read fixed content
    try:
        with open(content_path, 'r') as f:
            content = f.read()
    except Exception as e:
        logger.error(f"Error reading fixed template content: {str(e)}")
        return False
    
    if not HAS_DUCKDB or db_path.endswith('.json'):
        # Use JSON-based storage
        json_db_path = db_path if db_path.endswith('.json') else db_path.replace('.duckdb', '.json')
        
        try:
            if not os.path.exists(json_db_path):
                logger.error(f"JSON database file not found: {json_db_path}")
                return False
            
            # Load the JSON database
            with open(json_db_path, 'r') as f:
                template_db = json.load(f)
            
            if 'templates' not in template_db:
                logger.error("No templates found in JSON database")
                return False
            
            templates = template_db['templates']
            
            # Find matching template
            if template_id not in templates:
                logger.error(f"Template not found: {template_id}")
                return False
            
            # Update template
            templates[template_id]['template'] = content
            templates[template_id]['updated_at'] = datetime.now().isoformat()
            
            # Save the updated database
            with open(json_db_path, 'w') as f:
                json.dump(template_db, f, indent=2)
            
            logger.info(f"Updated template {template_id} in JSON database")
            return True
        
        except Exception as e:
            logger.error(f"Error saving template to JSON database: {str(e)}")
            return False
    
    else:
        # Use DuckDB
        try:
            import duckdb
            from datetime import datetime
            
            if not os.path.exists(db_path):
                logger.error(f"Database file not found: {db_path}")
                return False
            
            # Connect to the database
            conn = duckdb.connect(db_path)
            
            # Check if templates table exists
            table_check = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='templates'").fetchall()
            if not table_check:
                logger.error("No 'templates' table found in database")
                return False
            
            # Check if template exists
            template_check = conn.execute("SELECT id FROM templates WHERE id = ?", [template_id]).fetchall()
            if not template_check:
                logger.error(f"Template not found: {template_id}")
                return False
            
            # Update template
            conn.execute("""
            UPDATE templates
            SET template = ?, updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """, [content, template_id])
            
            conn.close()
            
            logger.info(f"Updated template {template_id} in DuckDB database")
            return True
        
        except Exception as e:
            logger.error(f"Error saving template to DuckDB database: {str(e)}")
            return False

def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(description="Template Extractor Tool")
    parser.add_argument("--extract-template", type=str, help="Extract a template by key (model_type/template_type[/platform])")
    parser.add_argument("--list-templates", action="store_true", help="List all templates in the database")
    parser.add_argument("--save-template", nargs=2, metavar=("TEMPLATE_ID", "CONTENT_PATH"), help="Save a fixed template to the database")
    parser.add_argument("--db-path", type=str, default="../generators/templates/template_db.json", help="Path to the database file")
    parser.add_argument("--output", type=str, help="Path to save extracted template")
    
    args = parser.parse_args()
    
    if args.list_templates:
        list_templates(args.db_path)
    elif args.extract_template:
        content = extract_template(args.db_path, args.extract_template)
        if content:
            if args.output:
                with open(args.output, 'w') as f:
                    f.write(content)
                print(f"Saved template to {args.output}")
            else:
                print(content)
    elif args.save_template:
        template_id, content_path = args.save_template
        success = save_template(args.db_path, template_id, content_path)
        if success:
            print(f"Successfully updated template {template_id}")
        else:
            print(f"Failed to update template {template_id}")
    else:
        parser.print_help()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())