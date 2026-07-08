#\!/usr/bin/env python3
"""
Create a DuckDB database for storing templates.

This script creates a DuckDB database for storing templates, migrating
them from the existing JSON-based storage.
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

try:
    import duckdb
    HAS_DUCKDB = True
except ImportError:
    HAS_DUCKDB = False
    print("DuckDB not available. Install with 'pip install duckdb' to continue.")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_template_db(db_path: str):
    """
    Create a DuckDB database for storing templates.
    
    Args:
        db_path: Path to create the database at
    """
    try:
        # Connect to database
        conn = duckdb.connect(db_path)
        
        # Create tables
        conn.execute("""
        CREATE TABLE IF NOT EXISTS templates (
            id VARCHAR PRIMARY KEY,
            model_type VARCHAR,
            template_type VARCHAR,
            platform VARCHAR,
            template TEXT,
            file_path VARCHAR,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        
        conn.execute("""
        CREATE TABLE IF NOT EXISTS template_metadata (
            id INTEGER PRIMARY KEY,
            version VARCHAR,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            description VARCHAR
        )
        """)
        
        # Insert initial metadata
        conn.execute("""
        INSERT INTO template_metadata (version, description)
        VALUES (?, ?)
        """, ["1.0.0", "Initial DuckDB template database"])
        
        # Create indices
        conn.execute("""
        CREATE INDEX IF NOT EXISTS template_model_type_idx ON templates(model_type)
        """)
        
        conn.execute("""
        CREATE INDEX IF NOT EXISTS template_template_type_idx ON templates(template_type)
        """)
        
        logger.info(f"Created DuckDB template database at {db_path}")
        
        # Commit changes and close connection
        conn.close()
        
        return True
    except Exception as e:
        logger.error(f"Error creating template database: {str(e)}")
        return False

def migrate_json_to_duckdb(json_path: str, db_path: str):
    """
    Migrate templates from JSON to DuckDB.
    
    Args:
        json_path: Path to the JSON file containing templates
        db_path: Path to the DuckDB database
    """
    try:
        # Load JSON data
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        if 'templates' not in data:
            logger.error("No templates found in JSON data")
            return False
        
        templates = data['templates']
        if not templates:
            logger.error("No templates found in JSON data")
            return False
        
        # Create database if it doesn't exist
        if not os.path.exists(db_path):
            logger.info(f"Creating DuckDB database at {db_path}")
            create_template_db(db_path)
        
        # Connect to database
        conn = duckdb.connect(db_path)
        
        # Delete existing templates
        conn.execute("DELETE FROM templates")
        
        # Insert templates
        for template_id, template_data in templates.items():
            conn.execute("""
            INSERT INTO templates (id, model_type, template_type, platform, template, file_path, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                template_id,
                template_data.get('model_type'),
                template_data.get('template_type'),
                template_data.get('platform'),
                template_data.get('template'),
                template_data.get('file_path'),
                datetime.now(),
                datetime.now()
            ])
        
        # Update metadata
        conn.execute("""
        INSERT INTO template_metadata (version, description)
        VALUES (?, ?)
        """, ["1.1.0", f"Migrated from JSON at {datetime.now().isoformat()}"])
        
        # Commit changes and close connection
        conn.close()
        
        logger.info(f"Migrated {len(templates)} templates from JSON to DuckDB")
        
        return True
    except Exception as e:
        logger.error(f"Error migrating JSON to DuckDB: {str(e)}")
        return False

def validate_duckdb_templates(db_path: str):
    """
    Validate templates in DuckDB database.
    
    Args:
        db_path: Path to the DuckDB database
    """
    try:
        # Connect to database
        conn = duckdb.connect(db_path)
        
        # Get all templates
        templates = conn.execute("SELECT id, model_type, template_type, platform, template FROM templates").fetchall()
        
        valid_count = 0
        invalid_templates = []
        
        import ast
        
        # Validate template syntax
        for template_id, model_type, template_type, platform, content in templates:
            try:
                ast.parse(content)
                valid_count += 1
            except SyntaxError:
                invalid_templates.append(template_id)
        
        logger.info(f"Found {valid_count}/{len(templates)} templates with valid syntax ({valid_count/len(templates)*100:.1f}%)")
        
        if invalid_templates:
            logger.warning(f"Found {len(invalid_templates)} templates with invalid syntax:")
            for template_id in invalid_templates[:10]:  # Show first 10
                logger.warning(f"  - {template_id}")
            
            if len(invalid_templates) > 10:
                logger.warning(f"  ... and {len(invalid_templates) - 10} more")
        
        # Close connection
        conn.close()
        
        return True
    except Exception as e:
        logger.error(f"Error validating DuckDB templates: {str(e)}")
        return False

def main():
    """Main function for standalone usage"""
    parser = argparse.ArgumentParser(description="Template Database Creator")
    parser.add_argument("--json-path", type=str, default="../generators/templates/template_db.json",
                      help="Path to JSON file containing templates")
    parser.add_argument("--db-path", type=str, default="../generators/templates/template_db.duckdb",
                      help="Path to create/use DuckDB database")
    parser.add_argument("--create-only", action="store_true",
                      help="Only create the database, don't migrate")
    parser.add_argument("--validate", action="store_true",
                      help="Validate templates after migration")
    
    args = parser.parse_args()
    
    if args.create_only:
        success = create_template_db(args.db_path)
    else:
        success = migrate_json_to_duckdb(args.json_path, args.db_path)
    
    if success and args.validate:
        validate_duckdb_templates(args.db_path)
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
