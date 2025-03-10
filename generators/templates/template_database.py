#!/usr/bin/env python3
"""
Template database system for model code generation.

This module provides a DuckDB-backed template system for generating model test files,
benchmarks, and skills. It provides centralized storage, version tracking, and
hardware-specific template customization.
"""

import os
import sys
import json
import logging
import argparse
import importlib
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Try to import duckdb (will be used if available, otherwise fallback to json)
try:
    import duckdb
    DUCKDB_AVAILABLE = True
    logger.info("DuckDB is available, will use database storage")
except ImportError:
    DUCKDB_AVAILABLE = False
    logger.warning("DuckDB not available, will use JSON file storage as fallback")

# Define common constants
DEFAULT_DB_PATH = os.path.join(os.path.dirname(__file__), "template_db.duckdb")
DEFAULT_JSON_PATH = os.path.join(os.path.dirname(__file__), "template_db.json")
TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), "model_templates")

# Model type definitions
MODEL_TYPES = [
    "bert", "t5", "llama", "vit", "clip", "whisper", "wav2vec2", 
    "clap", "llava", "xclip", "qwen", "detr", "default", "text_embedding",
    "vision", "video", "qwen2"
]

# Hardware platform definitions
HARDWARE_PLATFORMS = [
    "cpu", "cuda", "rocm", "mps", "openvino", "qualcomm", "samsung", "webnn", "webgpu"
]

# Template types
TEMPLATE_TYPES = [
    "test", "benchmark", "skill", "helper", "hardware_specific"
]

class TemplateDatabase:
    """Template database manager for model code generation."""

    def __init__(self, db_path: str = DEFAULT_DB_PATH, json_path: str = DEFAULT_JSON_PATH):
        """Initialize template database manager.
        
        Args:
            db_path: Path to DuckDB database file
            json_path: Path to JSON fallback file (used if DuckDB not available)
        """
        self.db_path = db_path
        self.json_path = json_path
        
        # Create connection to database if available
        self.conn = None
        if DUCKDB_AVAILABLE:
            try:
                self.conn = duckdb.connect(db_path)
                # Create table if it doesn't exist
                self._ensure_db_schema()
            except Exception as e:
                logger.error(f"Error connecting to database: {e}")
                self.conn = None
        
        # Load JSON fallback if DuckDB not available
        self.json_templates = None
        if not DUCKDB_AVAILABLE or self.conn is None:
            if os.path.exists(json_path):
                try:
                    with open(json_path, 'r') as f:
                        self.json_templates = json.load(f)
                except Exception as e:
                    logger.error(f"Error loading JSON templates: {e}")
                    self.json_templates = {}
            else:
                logger.warning(f"JSON template file not found: {json_path}")
                self.json_templates = {}
    
    def __del__(self):
        """Clean up database connection."""
        if self.conn is not None:
            self.conn.close()
    
    def _ensure_db_schema(self):
        """Ensure database schema exists."""
        if self.conn is None:
            return False
        
        try:
            # Check if templates table exists
            result = self.conn.execute("""
            SELECT name FROM sqlite_master
            WHERE type='table' AND name='templates'
            """).fetchone()
            
            if not result:
                # Create templates table
                self.conn.execute("""
                CREATE TABLE templates (
                    model_type VARCHAR,
                    template_type VARCHAR,
                    template TEXT,
                    hardware_platform VARCHAR,
                    created_at TIMESTAMP,
                    updated_at TIMESTAMP,
                    version INTEGER DEFAULT 1
                )
                """)
                logger.info("Created templates table")
            
            return True
        except Exception as e:
            logger.error(f"Error ensuring database schema: {e}")
            return False
    
    def get_template(self, model_type: str, template_type: str, 
                    hardware_platform: Optional[str] = None) -> Optional[str]:
        """Get a template from the database.
        
        Uses a fallback mechanism:
        1. Try hardware-specific template for the model type
        2. Fall back to generic template for the model type
        3. Fall back to hardware-specific default template
        4. Fall back to generic default template
        
        Args:
            model_type: Model type (bert, t5, llama, etc.)
            template_type: Template type (test, benchmark, skill, etc.)
            hardware_platform: Optional hardware platform (cuda, rocm, etc.)
            
        Returns:
            Template string or None if not found
        """
        # Try DuckDB first if available
        if self.conn is not None:
            try:
                template = self._get_template_from_db(model_type, template_type, hardware_platform)
                if template:
                    return template
            except Exception as e:
                logger.error(f"Error getting template from database: {e}")
        
        # Fall back to JSON if available
        if self.json_templates is not None:
            return self._get_template_from_json(model_type, template_type, hardware_platform)
        
        # Nothing worked, return None
        logger.error(f"No template found for {model_type}/{template_type}/{hardware_platform}")
        return None
    
    def _get_template_from_db(self, model_type: str, template_type: str, 
                             hardware_platform: Optional[str] = None) -> Optional[str]:
        """Get a template from the DuckDB database."""
        # Query for hardware-specific template first if hardware_platform provided
        if hardware_platform:
            result = self.conn.execute("""
            SELECT template FROM templates
            WHERE model_type = ? AND template_type = ? AND hardware_platform = ?
            ORDER BY version DESC LIMIT 1
            """, [model_type, template_type, hardware_platform]).fetchone()
            
            if result:
                return result[0]
        
        # Fall back to generic template for the model type
        result = self.conn.execute("""
        SELECT template FROM templates
        WHERE model_type = ? AND template_type = ? AND (hardware_platform IS NULL OR hardware_platform = '')
        ORDER BY version DESC LIMIT 1
        """, [model_type, template_type]).fetchone()
        
        if result:
            return result[0]
        
        # Fall back to default template type
        if hardware_platform:
            # Try hardware-specific default
            result = self.conn.execute("""
            SELECT template FROM templates
            WHERE model_type = 'default' AND template_type = ? AND hardware_platform = ?
            ORDER BY version DESC LIMIT 1
            """, [template_type, hardware_platform]).fetchone()
            
            if result:
                return result[0]
        
        # Fall back to generic default
        result = self.conn.execute("""
        SELECT template FROM templates
        WHERE model_type = 'default' AND template_type = ? AND (hardware_platform IS NULL OR hardware_platform = '')
        ORDER BY version DESC LIMIT 1
        """, [template_type]).fetchone()
        
        return result[0] if result else None
    
    def _get_template_from_json(self, model_type: str, template_type: str, 
                               hardware_platform: Optional[str] = None) -> Optional[str]:
        """Get a template from the JSON fallback."""
        # Try hardware-specific template
        if hardware_platform and model_type in self.json_templates:
            key = f"{template_type}_{hardware_platform}"
            if key in self.json_templates[model_type]:
                return self.json_templates[model_type][key]
        
        # Try generic template
        if model_type in self.json_templates and template_type in self.json_templates[model_type]:
            return self.json_templates[model_type][template_type]
        
        # Try default templates
        if hardware_platform and 'default' in self.json_templates:
            key = f"{template_type}_{hardware_platform}"
            if key in self.json_templates['default']:
                return self.json_templates['default'][key]
        
        # Try generic default
        if 'default' in self.json_templates and template_type in self.json_templates['default']:
            return self.json_templates['default'][template_type]
        
        return None
    
    def add_template(self, model_type: str, template_type: str, template: str, 
                    hardware_platform: Optional[str] = None) -> bool:
        """Add a new template to the database.
        
        Args:
            model_type: Model type (bert, t5, llama, etc.)
            template_type: Template type (test, benchmark, skill, etc.)
            template: Template content
            hardware_platform: Optional hardware platform (cuda, rocm, etc.)
            
        Returns:
            True if successful, False otherwise
        """
        # Try to add to DuckDB if available
        if self.conn is not None:
            try:
                # Check if template exists and get current version
                result = self.conn.execute("""
                SELECT version FROM templates
                WHERE model_type = ? AND template_type = ? AND 
                     (hardware_platform IS ? OR (hardware_platform IS NULL AND ? IS NULL))
                ORDER BY version DESC LIMIT 1
                """, [model_type, template_type, hardware_platform, hardware_platform]).fetchone()
                
                version = 1  # Default for new templates
                if result:
                    version = result[0] + 1
                
                now = datetime.now()
                self.conn.execute("""
                INSERT INTO templates 
                (model_type, template_type, template, hardware_platform, created_at, updated_at, version)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """, [model_type, template_type, template, hardware_platform, now, now, version])
                
                logger.info(f"Added template: {model_type}/{template_type}/{hardware_platform} (v{version})")
                return True
            except Exception as e:
                logger.error(f"Error adding template to database: {e}")
        
        # Fall back to JSON
        if self.json_templates is not None:
            if model_type not in self.json_templates:
                self.json_templates[model_type] = {}
            
            key = template_type
            if hardware_platform:
                key = f"{template_type}_{hardware_platform}"
            
            self.json_templates[model_type][key] = template
            
            try:
                with open(self.json_path, 'w') as f:
                    json.dump(self.json_templates, f, indent=2)
                
                logger.info(f"Added template to JSON: {model_type}/{key}")
                return True
            except Exception as e:
                logger.error(f"Error adding template to JSON: {e}")
                return False
        
        # Nothing worked
        return False
    
    def list_templates(self) -> List[Dict[str, Any]]:
        """List all templates in the database.
        
        Returns:
            List of template metadata
        """
        templates = []
        
        # Try DuckDB first if available
        if self.conn is not None:
            try:
                results = self.conn.execute("""
                SELECT model_type, template_type, hardware_platform, version, created_at, updated_at
                FROM templates
                ORDER BY model_type, template_type, hardware_platform, version DESC
                """).fetchall()
                
                for row in results:
                    model_type, template_type, hardware_platform, version, created_at, updated_at = row
                    templates.append({
                        'model_type': model_type,
                        'template_type': template_type,
                        'hardware_platform': hardware_platform or 'all',
                        'version': version,
                        'created_at': created_at,
                        'updated_at': updated_at
                    })
                
                return templates
            except Exception as e:
                logger.error(f"Error listing templates from database: {e}")
        
        # Fall back to JSON
        if self.json_templates is not None:
            for model_type, model_templates in self.json_templates.items():
                for key, _ in model_templates.items():
                    # Parse hardware platform from key if present
                    parts = key.split('_')
                    template_type = parts[0]
                    hardware_platform = '_'.join(parts[1:]) if len(parts) > 1 else 'all'
                    
                    templates.append({
                        'model_type': model_type,
                        'template_type': template_type,
                        'hardware_platform': hardware_platform,
                        'version': 1,
                        'created_at': None,
                        'updated_at': None
                    })
        
        return templates
    
    def validate_template(self, template: str) -> Tuple[bool, Optional[str]]:
        """Validate a template by checking for Python syntax errors.
        
        Args:
            template: Template string to validate
            
        Returns:
            (True, None) if valid, (False, error_message) if invalid
        """
        try:
            # Create temporary file
            with tempfile.NamedTemporaryFile(suffix='.py', delete=False) as tmp:
                tmp.write(template.encode('utf-8'))
                tmp_path = tmp.name
            
            # Validate syntax
            with open(tmp_path, 'r') as f:
                compile(f.read(), tmp_path, 'exec')
            
            # Clean up
            os.unlink(tmp_path)
            
            return (True, None)
        except Exception as e:
            # Clean up
            if 'tmp_path' in locals():
                try:
                    os.unlink(tmp_path)
                except:
                    pass
            
            return (False, str(e))
    
    def validate_all_templates(self) -> Dict[str, Any]:
        """Validate all templates in the database.
        
        Returns:
            Dictionary with validation results
        """
        results = {
            'total': 0,
            'valid': 0,
            'invalid': 0,
            'errors': []
        }
        
        # Process DuckDB templates
        if self.conn is not None:
            try:
                templates = self.conn.execute("""
                SELECT model_type, template_type, hardware_platform, template, version
                FROM templates
                WHERE template IS NOT NULL AND template != ''
                """).fetchall()
                
                for row in templates:
                    model_type, template_type, hardware_platform, template, version = row
                    results['total'] += 1
                    
                    valid, error = self.validate_template(template)
                    if valid:
                        results['valid'] += 1
                    else:
                        results['invalid'] += 1
                        results['errors'].append({
                            'model_type': model_type,
                            'template_type': template_type,
                            'hardware_platform': hardware_platform or 'all',
                            'version': version,
                            'error': error
                        })
            except Exception as e:
                logger.error(f"Error validating database templates: {e}")
        
        # Process JSON templates
        elif self.json_templates is not None:
            for model_type, model_templates in self.json_templates.items():
                for key, template in model_templates.items():
                    results['total'] += 1
                    
                    # Parse hardware platform from key if present
                    parts = key.split('_')
                    template_type = parts[0]
                    hardware_platform = '_'.join(parts[1:]) if len(parts) > 1 else 'all'
                    
                    valid, error = self.validate_template(template)
                    if valid:
                        results['valid'] += 1
                    else:
                        results['invalid'] += 1
                        results['errors'].append({
                            'model_type': model_type,
                            'template_type': template_type,
                            'hardware_platform': hardware_platform,
                            'version': 1,
                            'error': error
                        })
        
        return results
    
    def render_template(self, template: str, variables: Dict[str, Any]) -> str:
        """Render a template with the given variables.
        
        Args:
            template: Template string
            variables: Dictionary of variables to substitute
            
        Returns:
            Rendered template
        """
        # Simple variable substitution using format
        result = template
        
        # Handle both {{var}} and {var} style variables
        for key, value in variables.items():
            result = result.replace(f"{{{{{key}}}}}", str(value))
            result = result.replace(f"{{{key}}}", str(value))
        
        return result
    
    def generate_test_file(self, model_type: str, model_name: str,
                          template_type: str = 'test',
                          hardware_platforms: Optional[List[str]] = None,
                          **variables) -> str:
        """Generate a test file from a template.
        
        Args:
            model_type: Model type (bert, t5, llama, etc.)
            model_name: Model name for the test
            template_type: Template type (default: 'test')
            hardware_platforms: List of hardware platforms to support
            **variables: Additional template variables
            
        Returns:
            Generated test file content
        """
        # Get hardware platforms to support
        if hardware_platforms is None:
            hardware_platforms = ['cpu', 'cuda']  # Default hardware platforms
        
        # Create variables dictionary
        template_vars = {
            'model_name': model_name,
            'normalized_name': ''.join(c if c.isalnum() else '_' for c in model_name),
            'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'model_type': model_type,
            **{f"has_{hw}": hw in hardware_platforms for hw in HARDWARE_PLATFORMS},
            **variables
        }
        
        # Get the template
        template = None
        
        # Try specialized template for each hardware platform
        # Here we look for hardware-specific templates that can handle multiple platforms
        for platform in hardware_platforms:
            template = self.get_template(model_type, template_type, platform)
            if template:
                break
        
        # Fall back to generic template
        if not template:
            template = self.get_template(model_type, template_type)
        
        # Fall back to default template
        if not template:
            template = self.get_template('default', template_type)
        
        if not template:
            logger.error(f"No template found for {model_type}/{template_type}")
            return f"# ERROR: No template found for {model_type}/{template_type}\n"
        
        # Render the template
        return self.render_template(template, template_vars)
    
    def export_to_json(self, json_path: Optional[str] = None) -> bool:
        """Export templates from database to JSON file.
        
        Args:
            json_path: Path to JSON file (default: instance json_path)
            
        Returns:
            True if successful, False otherwise
        """
        if json_path is None:
            json_path = self.json_path
        
        if self.conn is None:
            logger.error("No database connection available")
            return False
        
        try:
            # Query templates
            results = self.conn.execute("""
            SELECT t1.model_type, t1.template_type, t1.template, t1.hardware_platform, t1.version
            FROM templates t1
            JOIN (
                SELECT model_type, template_type, hardware_platform, MAX(version) as max_version
                FROM templates
                GROUP BY model_type, template_type, hardware_platform
            ) t2
            ON t1.model_type = t2.model_type AND 
               t1.template_type = t2.template_type AND 
               (t1.hardware_platform IS t2.hardware_platform OR (t1.hardware_platform IS NULL AND t2.hardware_platform IS NULL)) AND
               t1.version = t2.max_version
            ORDER BY t1.model_type, t1.template_type, t1.hardware_platform
            """).fetchall()
            
            # Organize templates by model type and template type
            templates_dict = {}
            for row in results:
                model_type, template_type, template, hardware, version = row
                
                if model_type not in templates_dict:
                    templates_dict[model_type] = {}
                
                # Handle hardware-specific templates
                if hardware:
                    key = f"{template_type}_{hardware}"
                else:
                    key = template_type
                    
                templates_dict[model_type][key] = template
            
            # Write to JSON file
            with open(json_path, 'w') as f:
                json.dump(templates_dict, f, indent=2)
            
            logger.info(f"Exported {len(results)} templates to {json_path}")
            return True
        except Exception as e:
            logger.error(f"Error exporting templates: {e}")
            return False
    
    def import_from_json(self, json_path: Optional[str] = None) -> bool:
        """Import templates from JSON file to database.
        
        Args:
            json_path: Path to JSON file (default: instance json_path)
            
        Returns:
            True if successful, False otherwise
        """
        if json_path is None:
            json_path = self.json_path
        
        if self.conn is None:
            logger.error("No database connection available")
            return False
        
        try:
            # Read JSON file
            with open(json_path, 'r') as f:
                templates_dict = json.load(f)
            
            # Import templates
            count = 0
            for model_type, templates in templates_dict.items():
                for key, template in templates.items():
                    # Parse template type and hardware platform
                    parts = key.split('_')
                    template_type = parts[0]
                    hardware = '_'.join(parts[1:]) if len(parts) > 1 else None
                    
                    # Validate the template
                    valid, error = self.validate_template(template)
                    if not valid:
                        logger.warning(f"Skipping invalid template {model_type}/{key}: {error}")
                        continue
                    
                    # Add to database
                    success = self.add_template(model_type, template_type, template, hardware)
                    if success:
                        count += 1
            
            logger.info(f"Imported {count} templates from {json_path}")
            return True
        except Exception as e:
            logger.error(f"Error importing templates: {e}")
            return False
    
    def import_from_directory(self, directory: str, model_type: Optional[str] = None) -> bool:
        """Import templates from a directory of Python files.
        
        Args:
            directory: Directory containing template files
            model_type: Model type to assign to templates (default: from filename)
            
        Returns:
            True if successful, False otherwise
        """
        if not os.path.isdir(directory):
            logger.error(f"Directory not found: {directory}")
            return False
        
        try:
            count = 0
            for root, _, files in os.walk(directory):
                for filename in files:
                    if not filename.endswith('.py'):
                        continue
                    
                    filepath = os.path.join(root, filename)
                    
                    # Parse model_type and template_type from filename
                    name_parts = filename.replace('.py', '').split('_')
                    if len(name_parts) < 2:
                        logger.warning(f"Skipping file with invalid name format: {filename}")
                        continue
                    
                    file_model_type = model_type or name_parts[0]
                    template_type = name_parts[1]
                    
                    # Check for hardware platform in filename
                    hardware = None
                    if len(name_parts) > 2 and name_parts[2] in HARDWARE_PLATFORMS:
                        hardware = name_parts[2]
                    
                    # Read template content
                    with open(filepath, 'r') as f:
                        template = f.read()
                    
                    # Validate the template
                    valid, error = self.validate_template(template)
                    if not valid:
                        logger.warning(f"Skipping invalid template {filepath}: {error}")
                        continue
                    
                    # Add to database
                    success = self.add_template(file_model_type, template_type, template, hardware)
                    if success:
                        count += 1
            
            logger.info(f"Imported {count} templates from {directory}")
            return True
        except Exception as e:
            logger.error(f"Error importing templates from directory: {e}")
            return False
    
    def get_supported_hardware_platforms(self, model_type: str, template_type: str) -> List[str]:
        """Get list of hardware platforms supported by a template.
        
        Args:
            model_type: Model type (bert, t5, llama, etc.)
            template_type: Template type (test, benchmark, skill, etc.)
            
        Returns:
            List of supported hardware platforms
        """
        supported = []
        
        # Try DuckDB first if available
        if self.conn is not None:
            try:
                # Check for hardware-specific templates
                results = self.conn.execute("""
                SELECT DISTINCT hardware_platform
                FROM templates
                WHERE model_type = ? AND template_type = ? AND hardware_platform IS NOT NULL
                """, [model_type, template_type]).fetchall()
                
                supported.extend([row[0] for row in results])
                
                # Check for generic template
                result = self.conn.execute("""
                SELECT COUNT(*)
                FROM templates
                WHERE model_type = ? AND template_type = ? AND (hardware_platform IS NULL OR hardware_platform = '')
                """, [model_type, template_type]).fetchone()
                
                if result and result[0] > 0:
                    # Generic template supports all hardware platforms
                    supported.extend([p for p in HARDWARE_PLATFORMS if p not in supported])
            except Exception as e:
                logger.error(f"Error getting supported hardware platforms from database: {e}")
        
        # Fall back to JSON
        elif self.json_templates is not None:
            if model_type in self.json_templates:
                # Check for hardware-specific templates
                for key in self.json_templates[model_type]:
                    parts = key.split('_')
                    if parts[0] == template_type and len(parts) > 1:
                        hardware = '_'.join(parts[1:])
                        if hardware in HARDWARE_PLATFORMS:
                            supported.append(hardware)
                
                # Check for generic template
                if template_type in self.json_templates[model_type]:
                    # Generic template supports all hardware platforms
                    supported.extend([p for p in HARDWARE_PLATFORMS if p not in supported])
        
        return supported

# Command-line interface
def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Template database manager for model code generation"
    )
    parser.add_argument(
        "--db-path", type=str, default=DEFAULT_DB_PATH,
        help=f"Path to template database file (default: {DEFAULT_DB_PATH})"
    )
    parser.add_argument(
        "--json-path", type=str, default=DEFAULT_JSON_PATH,
        help=f"Path to JSON fallback file (default: {DEFAULT_JSON_PATH})"
    )
    
    # Actions
    group = parser.add_argument_group("Actions")
    group.add_argument(
        "--list-templates", action="store_true",
        help="List all templates in the database"
    )
    group.add_argument(
        "--create-template", action="store_true",
        help="Create a new template"
    )
    group.add_argument(
        "--validate-templates", action="store_true",
        help="Validate all templates in the database"
    )
    group.add_argument(
        "--export", action="store_true",
        help="Export templates from database to JSON file"
    )
    group.add_argument(
        "--import", action="store_true", dest="import_json",
        help="Import templates from JSON file to database"
    )
    group.add_argument(
        "--import-directory", type=str,
        help="Import templates from a directory of Python files"
    )
    group.add_argument(
        "--generate-test", action="store_true",
        help="Generate a test file from a template"
    )
    
    # Template creation arguments
    group = parser.add_argument_group("Template Creation")
    group.add_argument(
        "--model-type", type=str,
        help="Model type (bert, t5, llama, etc.)"
    )
    group.add_argument(
        "--template-type", type=str, default="test",
        help="Template type (test, benchmark, skill, etc.)"
    )
    group.add_argument(
        "--hardware-platform", type=str,
        help="Hardware platform (cuda, rocm, etc.)"
    )
    group.add_argument(
        "--template-file", type=str,
        help="File containing template content"
    )
    group.add_argument(
        "--store-in-db", action="store_true",
        help="Store template in database (with --create-template)"
    )
    
    # Test generation arguments
    group = parser.add_argument_group("Test Generation")
    group.add_argument(
        "--model-name", type=str,
        help="Model name for test generation"
    )
    group.add_argument(
        "--hardware-platforms", type=str,
        help="Comma-separated list of hardware platforms to support in test"
    )
    group.add_argument(
        "--output", type=str,
        help="Output file for generated test (default: stdout)"
    )
    
    # Other arguments
    group = parser.add_argument_group("Other")
    group.add_argument(
        "--verbose", action="store_true",
        help="Enable verbose logging"
    )
    group.add_argument(
        "--update-templates", action="store_true",
        help="Update all templates to latest versions"
    )
    
    return parser.parse_args()

def main():
    """Command-line interface main function."""
    args = parse_args()
    
    # Configure logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)
    
    # Create template database manager
    db = TemplateDatabase(args.db_path, args.json_path)
    
    # Process commands
    if args.list_templates:
        templates = db.list_templates()
        
        print("\nAvailable templates:")
        print("-" * 80)
        print(f"{'Model Type':<15} {'Template Type':<15} {'Hardware':<10} {'Version':<8}")
        print("-" * 80)
        
        for template in templates:
            model_type = template['model_type']
            template_type = template['template_type']
            hardware = template['hardware_platform'] or 'all'
            version = template['version']
            
            print(f"{model_type:<15} {template_type:<15} {hardware:<10} {version:<8}")
    
    elif args.create_template:
        if not args.model_type:
            logger.error("--model-type is required for template creation")
            return 1
        
        if not args.template_file:
            logger.error("--template-file is required for template creation")
            return 1
        
        # Read template content
        try:
            with open(args.template_file, 'r') as f:
                template = f.read()
        except Exception as e:
            logger.error(f"Error reading template file: {e}")
            return 1
        
        # Validate template
        valid, error = db.validate_template(template)
        if not valid:
            logger.error(f"Template validation failed: {error}")
            return 1
        
        # Store in database if requested
        if args.store_in_db:
            success = db.add_template(
                args.model_type,
                args.template_type,
                template,
                args.hardware_platform
            )
            
            if not success:
                logger.error("Failed to store template in database")
                return 1
            
            logger.info(f"Template stored successfully: {args.model_type}/{args.template_type}/{args.hardware_platform or 'all'}")
        else:
            # Just print template info
            logger.info(f"Template validated successfully: {args.template_file}")
            print(f"Model Type: {args.model_type}")
            print(f"Template Type: {args.template_type}")
            print(f"Hardware Platform: {args.hardware_platform or 'all'}")
    
    elif args.validate_templates:
        results = db.validate_all_templates()
        
        print("\nTemplate Validation Results:")
        print("-" * 80)
        print(f"Total templates: {results['total']}")
        print(f"Valid templates: {results['valid']} ({results['valid'] / results['total'] * 100:.1f}%)")
        print(f"Invalid templates: {results['invalid']}")
        
        if results['invalid'] > 0:
            print("\nInvalid templates:")
            for error in results['errors']:
                print(f"\n{error['model_type']}/{error['template_type']}/{error['hardware_platform']} (v{error['version']}):")
                print(f"  Error: {error['error']}")
        
        return 0 if results['invalid'] == 0 else 1
    
    elif args.export:
        success = db.export_to_json(args.json_path)
        if not success:
            logger.error("Failed to export templates")
            return 1
    
    elif args.import_json:
        success = db.import_from_json(args.json_path)
        if not success:
            logger.error("Failed to import templates")
            return 1
    
    elif args.import_directory:
        success = db.import_from_directory(args.import_directory, args.model_type)
        if not success:
            logger.error("Failed to import templates from directory")
            return 1
    
    elif args.generate_test:
        if not args.model_type:
            logger.error("--model-type is required for test generation")
            return 1
        
        if not args.model_name:
            logger.error("--model-name is required for test generation")
            return 1
        
        # Parse hardware platforms
        hardware_platforms = None
        if args.hardware_platforms:
            hardware_platforms = args.hardware_platforms.split(',')
        
        # Generate test file
        test_content = db.generate_test_file(
            args.model_type,
            args.model_name,
            args.template_type,
            hardware_platforms
        )
        
        # Output test file
        if args.output:
            try:
                with open(args.output, 'w') as f:
                    f.write(test_content)
                logger.info(f"Test file written to {args.output}")
            except Exception as e:
                logger.error(f"Error writing test file: {e}")
                return 1
        else:
            print(test_content)
    
    elif args.update_templates:
        if not args.model_type:
            logger.error("--model-type is required for template update")
            return 1
        
        print(f"Updating templates for model type {args.model_type}")
        # TODO: Implement template update
        logger.warning("Template update not implemented yet")
    
    else:
        logger.error("No action specified")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())