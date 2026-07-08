#!/usr/bin/env python3
"""
Template database utilities.

This module provides utilities for working with template databases, including:
- Database connection and schema management
- Template retrieval, creation, and updating
- Validation status tracking
- Inheritance relationship management
- Hardware compatibility management
"""

import os
import json
import logging
import datetime
from typing import Dict, Any, List, Tuple, Optional, Set, Union

from .placeholder_helpers import get_modality_for_model_type
from .template_validation import validate_template, validate_template_db_schema
from .template_inheritance import get_parent_for_model_type, get_default_parent_templates

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = "./template_db.duckdb"

def get_db_connection(db_path: str = DEFAULT_DB_PATH):
    """
    Get a DuckDB connection to the template database
    
    Args:
        db_path (str): Path to the database file
        
    Returns:
        duckdb.DuckDBPyConnection: Database connection
    """
    try:
        import duckdb
        conn = duckdb.connect(db_path)
        return conn
    except ImportError:
        logger.error("DuckDB is not installed. Please install it with: pip install duckdb")
        raise ImportError("DuckDB is required for template database operations")
    except Exception as e:
        logger.error(f"Error connecting to database: {e}")
        raise

def create_schema(conn) -> bool:
    """
    Create the template database schema if it doesn't exist
    
    Args:
        conn: DuckDB connection
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Create templates table
        conn.execute("""
        CREATE TABLE IF NOT EXISTS templates (
            id INTEGER PRIMARY KEY,
            model_type VARCHAR NOT NULL,
            template_type VARCHAR NOT NULL,
            template TEXT NOT NULL,
            hardware_platform VARCHAR,
            validation_status VARCHAR,
            parent_template VARCHAR,
            modality VARCHAR,
            last_updated TIMESTAMP
        )
        """)
        
        # Create template_validation table
        conn.execute("""
        CREATE TABLE IF NOT EXISTS template_validation (
            id INTEGER PRIMARY KEY,
            template_id INTEGER,
            validation_date TIMESTAMP,
            validation_type VARCHAR,
            success BOOLEAN,
            errors TEXT,
            hardware_support TEXT,
            FOREIGN KEY (template_id) REFERENCES templates(id)
        )
        """)
        
        # Create template_placeholders table
        conn.execute("""
        CREATE TABLE IF NOT EXISTS template_placeholders (
            id INTEGER PRIMARY KEY,
            placeholder VARCHAR NOT NULL,
            description TEXT,
            default_value VARCHAR,
            required BOOLEAN
        )
        """)
        
        logger.info("Template database schema created successfully")
        return True
    except Exception as e:
        logger.error(f"Error creating database schema: {e}")
        return False

def check_database(db_path: str = DEFAULT_DB_PATH) -> bool:
    """
    Check if database exists and has the correct schema
    
    Args:
        db_path (str): Path to the database file
        
    Returns:
        bool: True if database exists and has correct schema, False otherwise
    """
    if not os.path.exists(db_path):
        logger.error(f"Database file {db_path} does not exist")
        return False
    
    try:
        conn = get_db_connection(db_path)
        
        # Check if tables exist
        result = conn.execute("""
        SELECT count(*) FROM information_schema.tables 
        WHERE table_name IN ('templates', 'template_validation', 'template_placeholders')
        """).fetchone()
        
        if result[0] != 3:
            logger.error("One or more required tables are missing from the database")
            conn.close()
            return False
        
        # Check if templates table has the expected columns
        result = conn.execute("""
        PRAGMA table_info(templates)
        """).fetchall()
        
        columns = [row[1] for row in result]
        
        # Validate columns
        success, missing_columns = validate_template_db_schema(columns)
        
        if not success:
            logger.error(f"Templates table is missing required columns: {missing_columns}")
            conn.close()
            return False
        
        # Check if database has templates
        result = conn.execute("""
        SELECT COUNT(*) FROM templates
        """).fetchone()
        
        template_count = result[0]
        if template_count == 0:
            logger.warning("Database exists but contains no templates")
        else:
            logger.info(f"Database contains {template_count} templates")
        
        conn.close()
        return True
    except Exception as e:
        logger.error(f"Error checking database: {e}")
        return False

def get_template(
    db_path: str, 
    model_type: str, 
    template_type: str, 
    hardware_platform: Optional[str] = None
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Get a template from the database with inheritance support
    
    Args:
        db_path (str): Path to the database file
        model_type (str): The model type to get template for
        template_type (str): The template type to get
        hardware_platform (Optional[str]): Specific hardware platform to get template for
        
    Returns:
        Tuple[Optional[str], Optional[str], Optional[str]]: 
            Template content, parent template name, and template modality
    """
    try:
        conn = get_db_connection(db_path)
        
        # Try to get a hardware-specific template first if requested
        if hardware_platform:
            query = """
            SELECT template, parent_template, modality FROM templates
            WHERE model_type = ? AND template_type = ? AND hardware_platform = ?
            """
            params = [model_type, template_type, hardware_platform]
        else:
            # Get generic template (hardware_platform is NULL or empty)
            query = """
            SELECT template, parent_template, modality FROM templates
            WHERE model_type = ? AND template_type = ? AND (hardware_platform IS NULL OR hardware_platform = '')
            """
            params = [model_type, template_type]
        
        result = conn.execute(query, params).fetchone()
        
        if result:
            template, parent_template, modality = result
            conn.close()
            return template, parent_template, modality
        
        # If no template found, try to find a parent template
        # First, determine parent template type from model type
        parent, modality = get_parent_for_model_type(model_type)
        
        if parent:
            logger.info(f"No template found for {model_type}, using parent template {parent}")
            
            # Try to get parent template
            query = """
            SELECT template FROM templates
            WHERE model_type = ? AND template_type = ? AND (hardware_platform IS NULL OR hardware_platform = '')
            """
            params = [parent, template_type]
            
            result = conn.execute(query, params).fetchone()
            
            if result:
                conn.close()
                return result[0], parent, modality
        
        # If no parent template found, check for a default template
        query = """
        SELECT template FROM templates
        WHERE model_type = 'default' AND template_type = ? AND (hardware_platform IS NULL OR hardware_platform = '')
        """
        params = [template_type]
        
        result = conn.execute(query, params).fetchone()
        
        if result:
            conn.close()
            return result[0], 'default', modality or 'unknown'
        
        # If all else fails, return None
        conn.close()
        logger.warning(f"No template found for {model_type}/{template_type}/{hardware_platform or 'generic'}")
        
        return None, None, modality or 'unknown'
    except Exception as e:
        logger.error(f"Error getting template: {e}")
        return None, None, 'unknown'

def store_template(
    db_path: str,
    model_type: str,
    template_type: str,
    template_content: str,
    hardware_platform: Optional[str] = None,
    parent_template: Optional[str] = None,
    modality: Optional[str] = None,
    validate: bool = True
) -> bool:
    """
    Store a template in the database
    
    Args:
        db_path (str): Path to the database file
        model_type (str): The model type to store template for
        template_type (str): The template type to store
        template_content (str): The template content
        hardware_platform (Optional[str]): Specific hardware platform the template is for
        parent_template (Optional[str]): Parent template name
        modality (Optional[str]): Template modality
        validate (bool): Whether to validate the template before storing
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        conn = get_db_connection(db_path)
        
        # Determine modality if not provided
        if not modality:
            modality = get_modality_for_model_type(model_type)
        
        # Determine parent template if not provided
        if not parent_template:
            parent_template, _ = get_parent_for_model_type(model_type)
        
        # Validate template if requested
        validation_status = "UNKNOWN"
        validation_results = None
        
        if validate:
            success, results = validate_template(
                template_content, template_type, model_type, hardware_platform
            )
            validation_status = "VALID" if success else "INVALID"
            validation_results = results
        
        # Check if template already exists
        if hardware_platform:
            result = conn.execute("""
            SELECT id FROM templates
            WHERE model_type = ? AND template_type = ? AND hardware_platform = ?
            """, [model_type, template_type, hardware_platform]).fetchone()
        else:
            result = conn.execute("""
            SELECT id FROM templates
            WHERE model_type = ? AND template_type = ? AND (hardware_platform IS NULL OR hardware_platform = '')
            """, [model_type, template_type]).fetchone()
        
        current_timestamp = datetime.datetime.now()
        
        if result:
            # Update existing template
            template_id = result[0]
            
            conn.execute("""
            UPDATE templates
            SET template = ?,
                parent_template = ?,
                modality = ?,
                validation_status = ?,
                last_updated = ?
            WHERE id = ?
            """, [
                template_content,
                parent_template,
                modality,
                validation_status,
                current_timestamp,
                template_id
            ])
            
            logger.info(f"Updated template for {model_type}/{template_type}/{hardware_platform or 'generic'}")
        else:
            # Insert new template
            conn.execute("""
            INSERT INTO templates
            (model_type, template_type, template, hardware_platform, validation_status, parent_template, modality, last_updated)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                model_type,
                template_type,
                template_content,
                hardware_platform,
                validation_status,
                parent_template,
                modality,
                current_timestamp
            ])
            
            # Get the new template ID
            result = conn.execute("SELECT last_insert_rowid()").fetchone()
            template_id = result[0]
            
            logger.info(f"Inserted new template for {model_type}/{template_type}/{hardware_platform or 'generic'}")
        
        # Store validation results if available
        if validation_results:
            # Convert hardware support dict to JSON string
            hardware_support_json = json.dumps(validation_results['hardware']['support'])
            
            # Convert errors to JSON string
            errors_json = json.dumps(validation_results['syntax']['errors'])
            
            conn.execute("""
            INSERT INTO template_validation
            (template_id, validation_date, validation_type, success, errors, hardware_support)
            VALUES (?, ?, ?, ?, ?, ?)
            """, [
                template_id,
                current_timestamp,
                'full',
                validation_status == "VALID",
                errors_json,
                hardware_support_json
            ])
        
        conn.close()
        return True
    except Exception as e:
        logger.error(f"Error storing template: {e}")
        return False

def list_templates(db_path: str = DEFAULT_DB_PATH, model_type: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    List all templates in the database with their validation status
    
    Args:
        db_path (str): Path to the database file
        model_type (Optional[str]): Filter templates by model type
        
    Returns:
        List[Dict[str, Any]]: List of template information dictionaries
    """
    try:
        conn = get_db_connection(db_path)
        
        # Query templates with validation status
        if model_type:
            query = """
            SELECT t.id, t.model_type, t.template_type, t.hardware_platform, 
                   t.validation_status, t.modality, t.parent_template,
                   v.validation_date, v.success as latest_validation,
                   v.hardware_support
            FROM templates t
            LEFT JOIN (
                SELECT template_id, MAX(validation_date) as validation_date
                FROM template_validation
                GROUP BY template_id
            ) latest ON t.id = latest.template_id
            LEFT JOIN template_validation v ON latest.template_id = v.template_id 
                AND latest.validation_date = v.validation_date
            WHERE t.model_type = ?
            ORDER BY t.model_type, t.template_type, t.hardware_platform
            """
            results = conn.execute(query, [model_type]).fetchall()
        else:
            query = """
            SELECT t.id, t.model_type, t.template_type, t.hardware_platform, 
                   t.validation_status, t.modality, t.parent_template,
                   v.validation_date, v.success as latest_validation,
                   v.hardware_support
            FROM templates t
            LEFT JOIN (
                SELECT template_id, MAX(validation_date) as validation_date
                FROM template_validation
                GROUP BY template_id
            ) latest ON t.id = latest.template_id
            LEFT JOIN template_validation v ON latest.template_id = v.template_id 
                AND latest.validation_date = v.validation_date
            ORDER BY t.model_type, t.template_type, t.hardware_platform
            """
            results = conn.execute(query).fetchall()
        
        templates = []
        for row in results:
            template_id, model_type, template_type, hardware_platform, status, modality, parent, \
            validation_date, latest_success, hardware_support = row
            
            # Format hardware platform display
            hardware = hardware_platform or "generic"
            
            # Format status display
            status = status or "UNKNOWN"
            
            # Format modality display
            modality = modality or "unknown"
            
            # Format validation date display
            validation_date_str = str(validation_date) if validation_date else "Never"
            
            # Format validation status
            if latest_success is not None:
                validation_status = "PASS" if latest_success else "FAIL"
            else:
                validation_status = "NONE"
            
            # Parse hardware support
            if hardware_support:
                hardware_info = json.loads(hardware_support)
                supported_hw = [hw for hw, supported in hardware_info.items() if supported]
                hw_display = ", ".join(supported_hw)
            else:
                hw_display = "Unknown"
            
            templates.append({
                "id": template_id,
                "model_type": model_type,
                "template_type": template_type,
                "hardware_platform": hardware,
                "status": status,
                "modality": modality,
                "parent_template": parent,
                "validation_date": validation_date_str,
                "validation_status": validation_status,
                "supported_hardware": hw_display
            })
        
        conn.close()
        return templates
    except Exception as e:
        logger.error(f"Error listing templates: {e}")
        return []

def add_default_parent_templates(db_path: str = DEFAULT_DB_PATH) -> bool:
    """
    Add default parent templates to the database
    
    Args:
        db_path (str): Path to the database file
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Get default parent templates
        default_parents = get_default_parent_templates()
        
        # Add each parent template
        for parent_type, templates in default_parents.items():
            modality = parent_type.replace('default_', '')
            
            for template_type, template_content in templates.items():
                store_template(
                    db_path=db_path,
                    model_type=parent_type,
                    template_type=template_type,
                    template_content=template_content,
                    hardware_platform=None,
                    parent_template=None,
                    modality=modality,
                    validate=True
                )
        
        logger.info("Default parent templates added successfully")
        return True
    except Exception as e:
        logger.error(f"Error adding default parent templates: {e}")
        return False

def update_template_inheritance(db_path: str = DEFAULT_DB_PATH) -> bool:
    """
    Update all templates in the database with inheritance information
    
    Args:
        db_path (str): Path to the database file
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        conn = get_db_connection(db_path)
        
        # Get all templates
        results = conn.execute("""
        SELECT id, model_type, template_type
        FROM templates
        """).fetchall()
        
        # Update each template with parent and modality information
        for template_id, model_type, template_type in results:
            # Skip default parent templates
            if model_type.startswith('default_'):
                continue
            
            # Determine parent template and modality
            parent, modality = get_parent_for_model_type(model_type)
            
            # Update template
            conn.execute("""
            UPDATE templates
            SET parent_template = ?,
                modality = ?,
                last_updated = ?
            WHERE id = ?
            """, [parent, modality, datetime.datetime.now(), template_id])
        
        conn.close()
        logger.info("Template inheritance updated successfully")
        return True
    except Exception as e:
        logger.error(f"Error updating template inheritance: {e}")
        return False

def validate_all_templates(db_path: str = DEFAULT_DB_PATH, model_type: Optional[str] = None) -> Dict[str, int]:
    """
    Validate all templates in the database
    
    Args:
        db_path (str): Path to the database file
        model_type (Optional[str]): Filter by model type
        
    Returns:
        Dict[str, int]: Dictionary with counts of valid and invalid templates
    """
    try:
        conn = get_db_connection(db_path)
        
        # Query templates to validate
        if model_type:
            query = """
            SELECT id, model_type, template_type, template, hardware_platform
            FROM templates
            WHERE model_type = ?
            """
            results = conn.execute(query, [model_type]).fetchall()
        else:
            query = """
            SELECT id, model_type, template_type, template, hardware_platform
            FROM templates
            """
            results = conn.execute(query).fetchall()
        
        if not results:
            logger.warning(f"No templates found to validate")
            return {"valid": 0, "invalid": 0, "total": 0}
        
        # Validate each template
        valid_count = 0
        invalid_count = 0
        
        for template_id, model_type, template_type, template, hardware_platform in results:
            logger.debug(f"Validating template: {model_type}/{template_type}/{hardware_platform or 'generic'}")
            
            # Validate template
            success, validation_results = validate_template(
                template, template_type, model_type, hardware_platform
            )
            
            # Update template with validation status
            status = "VALID" if success else "INVALID"
            if success:
                valid_count += 1
            else:
                invalid_count += 1
                
                # Log validation errors
                if not validation_results['syntax']['success']:
                    logger.error(f"Syntax errors in {model_type}/{template_type}: {validation_results['syntax']['errors']}")
                
                if not validation_results['placeholders']['success']:
                    logger.error(f"Missing placeholders in {model_type}/{template_type}: {validation_results['placeholders']['missing']}")
            
            # Update template validation status in database
            conn.execute("""
            UPDATE templates 
            SET validation_status = ?, 
                last_updated = ?
            WHERE id = ?
            """, [status, datetime.datetime.now(), template_id])
            
            # Store detailed validation results in template_validation table
            hardware_support_json = json.dumps(validation_results['hardware']['support'])
            conn.execute("""
            INSERT INTO template_validation
            (template_id, validation_date, validation_type, success, errors, hardware_support)
            VALUES (?, ?, ?, ?, ?, ?)
            """, [
                template_id, 
                datetime.datetime.now(), 
                'full', 
                success, 
                json.dumps(validation_results['syntax']['errors']), 
                hardware_support_json
            ])
        
        conn.close()
        logger.info(f"Validation complete: {valid_count} valid, {invalid_count} invalid")
        return {"valid": valid_count, "invalid": invalid_count, "total": valid_count + invalid_count}
    except Exception as e:
        logger.error(f"Error validating templates: {e}")
        return {"valid": 0, "invalid": 0, "total": 0}