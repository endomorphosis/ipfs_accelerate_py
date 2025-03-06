"""
DuckDB API implementation for hardware test templates.

This module replaces the previous JSON storage with a DuckDB database-based approach
for storing hardware test templates. It provides functions to:
1. Load templates from database
2. Store templates to database 
3. Query templates based on various parameters

Usage:
    from hardware_test_templates.template_database import TemplateDatabase
    
    # Initialize the database
    db = TemplateDatabase()
    
    # Get a template
    template = db.get_template("vit")
    
    # Store a template
    db.store_template("new_model", template_content)
"""

import os
import logging
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Union

try:
    import duckdb
    import pandas as pd
except ImportError:
    print("Error: Required packages not installed. Please install with:")
    print("pip install duckdb pandas")
    sys.exit(1)

# Add parent directory to path for module imports
sys.path.append(str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TemplateDatabase:
    """Database API for hardware test templates."""
    
    def __init__(self, db_path: str = "./template_db.duckdb", debug: bool = False):
        """
        Initialize the template database API.
        
        Args:
            db_path: Path to the DuckDB database
            debug: Enable debug logging
        """
        self.db_path = db_path
        
        # Set up logging
        if debug:
            logger.setLevel(logging.DEBUG)
        
        # Ensure database exists
        self._ensure_db_exists()
        
        # Convert legacy JSON to database if needed
        self._check_legacy_json()
        
        logger.info(f"Initialized TemplateDatabase with DB: {db_path}")
    
    def _ensure_db_exists(self):
        """
        Ensure that the database exists and has the expected schema.
        If not, initialize it with the schema creation script.
        """
        db_file = Path(self.db_path)
        
        # Check if parent directories exist
        if not db_file.parent.exists():
            db_file.parent.mkdir(parents=True, exist_ok=True)
        
        conn = self._get_connection()
        
        try:
            # Check if template table exists
            table_exists = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='model_templates'"
            ).fetchone()
            
            if not table_exists:
                logger.info("Creating template database schema")
                self._create_schema(conn)
        except Exception as e:
            logger.error(f"Error checking schema: {e}")
            self._create_schema(conn)
        finally:
            conn.close()
    
    def _create_schema(self, conn):
        """
        Create the database schema.
        
        Args:
            conn: Database connection
        """
        # Create model templates table
        conn.execute("""
        CREATE TABLE IF NOT EXISTS model_templates (
            template_id INTEGER PRIMARY KEY,
            model_id VARCHAR NOT NULL UNIQUE,
            model_name VARCHAR,
            model_family VARCHAR,
            modality VARCHAR,
            template_content TEXT NOT NULL,
            hardware_support JSON,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        
        # Create hardware platforms table
        conn.execute("""
        CREATE TABLE IF NOT EXISTS hardware_platforms (
            hardware_id INTEGER PRIMARY KEY,
            hardware_type VARCHAR NOT NULL UNIQUE,
            display_name VARCHAR,
            description TEXT,
            status VARCHAR
        )
        """)
        
        # Insert standard hardware platforms
        hardware_data = [
            (1, 'cpu', 'CPU', 'Standard CPU implementation', 'supported'),
            (2, 'cuda', 'CUDA', 'NVIDIA GPU implementation', 'supported'),
            (3, 'rocm', 'ROCm', 'AMD GPU implementation', 'supported'),
            (4, 'mps', 'MPS', 'Apple Silicon GPU implementation', 'supported'),
            (5, 'openvino', 'OpenVINO', 'Intel hardware acceleration', 'supported'),
            (6, 'webnn', 'WebNN', 'Web Neural Network API (browser)', 'supported'),
            (7, 'webgpu', 'WebGPU', 'Web GPU API (browser)', 'supported'),
            (8, 'webgpu_compute', 'WebGPU Compute', 'WebGPU with Compute Shaders', 'supported'),
            (9, 'webnn_parallelized', 'WebNN Parallelized', 'WebNN with parallel loading', 'supported'),
            (10, 'webgpu_parallelized', 'WebGPU Parallelized', 'WebGPU with parallel loading', 'supported'),
            (11, 'qualcomm', 'Qualcomm', 'Qualcomm AI Engine/Hexagon DSP implementation', 'supported')
        ]
        
        # Convert to DataFrame and insert
        hardware_df = pd.DataFrame(hardware_data, columns=[
            'hardware_id', 'hardware_type', 'display_name', 'description', 'status'
        ])
        
        conn.execute("INSERT OR IGNORE INTO hardware_platforms SELECT * FROM hardware_df")
        
        logger.info("Schema created successfully")
    
    def _check_legacy_json(self):
        """
        Check if legacy JSON file exists and migrate it to the database.
        """
        # Check if the original JSON file exists in the same directory
        json_path = Path(__file__).parent / 'template_database.json.bak'
        if not json_path.exists():
            json_path = Path(__file__).parent / 'template_database.json.original'
        
        if json_path.exists():
            logger.info(f"Found legacy JSON file: {json_path}")
            
            try:
                import json
                with open(json_path, 'r') as f:
                    templates = json.load(f)
                
                # Migrate each template to the database
                for model_id, template_content in templates.items():
                    if not self.get_template(model_id):
                        model_name = model_id
                        model_family = self._infer_model_family(model_id)
                        modality = self._infer_modality(model_id)
                        
                        self.store_template(
                            model_id=model_id,
                            template_content=template_content,
                            model_name=model_name,
                            model_family=model_family,
                            modality=modality
                        )
                
                logger.info(f"Migrated {len(templates)} templates from legacy JSON")
            except Exception as e:
                logger.error(f"Error migrating legacy JSON: {e}")
    
    def _infer_model_family(self, model_id: str) -> str:
        """Infer model family from model ID."""
        model_id_lower = model_id.lower()
        
        if 'bert' in model_id_lower:
            return 'bert'
        elif 't5' in model_id_lower:
            return 't5'
        elif 'gpt' in model_id_lower or 'llama' in model_id_lower:
            return 'llm'
        elif 'clip' in model_id_lower or 'vit' in model_id_lower:
            return 'vision'
        elif 'whisper' in model_id_lower or 'wav2vec' in model_id_lower or 'clap' in model_id_lower:
            return 'audio'
        elif 'llava' in model_id_lower:
            return 'multimodal'
        elif 'embedding' in model_id_lower:
            return 'embedding'
        else:
            return 'unknown'
    
    def _infer_modality(self, model_id: str) -> str:
        """Infer modality from model ID."""
        model_id_lower = model_id.lower()
        
        if 'bert' in model_id_lower or 't5' in model_id_lower or 'gpt' in model_id_lower or 'llama' in model_id_lower:
            return 'text'
        elif 'clip' in model_id_lower or 'vit' in model_id_lower or 'detr' in model_id_lower:
            return 'image'
        elif 'whisper' in model_id_lower or 'wav2vec' in model_id_lower or 'clap' in model_id_lower:
            return 'audio'
        elif 'video' in model_id_lower or 'xclip' in model_id_lower:
            return 'video'
        elif 'llava' in model_id_lower or 'vision_language' in model_id_lower:
            return 'multimodal'
        else:
            return 'unknown'
    
    def _get_connection(self):
        """Get a connection to the database."""
        return duckdb.connect(self.db_path)
    
    def get_template(self, model_id: str) -> Optional[str]:
        """
        Get a template by model ID.
        
        Args:
            model_id: ID of the model template to retrieve
            
        Returns:
            Template content or None if not found
        """
        conn = self._get_connection()
        try:
            result = conn.execute(
                "SELECT template_content FROM model_templates WHERE model_id = ?",
                [model_id]
            ).fetchone()
            
            if result:
                return result[0]
            else:
                logger.warning(f"Template not found for model_id: {model_id}")
                return None
        except Exception as e:
            logger.error(f"Error retrieving template: {e}")
            return None
        finally:
            conn.close()
    
    def store_template(self, model_id: str, template_content: str, model_name: Optional[str] = None,
                      model_family: Optional[str] = None, modality: Optional[str] = None,
                      hardware_support: Optional[Dict[str, Any]] = None) -> bool:
        """
        Store a template in the database.
        
        Args:
            model_id: ID of the model template
            template_content: Template content (Python code as string)
            model_name: Name of the model (defaults to model_id)
            model_family: Family of the model (bert, t5, etc.)
            modality: Modality of the model (text, image, audio, etc.)
            hardware_support: Dictionary of hardware support information
            
        Returns:
            True if successful, False otherwise
        """
        if not model_name:
            model_name = model_id
            
        if not model_family:
            model_family = self._infer_model_family(model_id)
            
        if not modality:
            modality = self._infer_modality(model_id)
        
        # Convert hardware support to JSON
        if hardware_support:
            try:
                import json
                hardware_support_json = json.dumps(hardware_support)
            except Exception as e:
                logger.error(f"Error converting hardware support to JSON: {e}")
                hardware_support_json = None
        else:
            hardware_support_json = None
        
        conn = self._get_connection()
        try:
            # Check if template exists
            exists = conn.execute(
                "SELECT COUNT(*) FROM model_templates WHERE model_id = ?",
                [model_id]
            ).fetchone()[0] > 0
            
            if exists:
                # Update existing template
                conn.execute(
                    """
                    UPDATE model_templates SET
                        template_content = ?,
                        model_name = ?,
                        model_family = ?,
                        modality = ?,
                        hardware_support = ?,
                        last_updated = CURRENT_TIMESTAMP
                    WHERE model_id = ?
                    """,
                    [template_content, model_name, model_family, modality, hardware_support_json, model_id]
                )
                logger.info(f"Updated template for model_id: {model_id}")
            else:
                # Get next template_id
                max_id = conn.execute("SELECT MAX(template_id) FROM model_templates").fetchone()[0]
                template_id = 1 if max_id is None else max_id + 1
                
                # Insert new template
                conn.execute(
                    """
                    INSERT INTO model_templates (
                        template_id, model_id, model_name, model_family, modality,
                        template_content, hardware_support
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    [template_id, model_id, model_name, model_family, modality, 
                     template_content, hardware_support_json]
                )
                logger.info(f"Inserted new template for model_id: {model_id}")
            
            return True
        except Exception as e:
            logger.error(f"Error storing template: {e}")
            return False
        finally:
            conn.close()
    
    def delete_template(self, model_id: str) -> bool:
        """
        Delete a template from the database.
        
        Args:
            model_id: ID of the model template to delete
            
        Returns:
            True if successful, False otherwise
        """
        conn = self._get_connection()
        try:
            conn.execute(
                "DELETE FROM model_templates WHERE model_id = ?",
                [model_id]
            )
            logger.info(f"Deleted template for model_id: {model_id}")
            return True
        except Exception as e:
            logger.error(f"Error deleting template: {e}")
            return False
        finally:
            conn.close()
    
    def list_templates(self, model_family: Optional[str] = None, 
                      modality: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List all templates, optionally filtered by model family or modality.
        
        Args:
            model_family: Filter by model family (optional)
            modality: Filter by modality (optional)
            
        Returns:
            List of template metadata dictionaries
        """
        sql = """
        SELECT
            model_id,
            model_name,
            model_family,
            modality,
            hardware_support,
            last_updated
        FROM
            model_templates
        """
        
        conditions = []
        parameters = []
        
        if model_family:
            conditions.append("model_family = ?")
            parameters.append(model_family)
        
        if modality:
            conditions.append("modality = ?")
            parameters.append(modality)
        
        if conditions:
            sql += " WHERE " + " AND ".join(conditions)
        
        sql += " ORDER BY model_family, model_id"
        
        conn = self._get_connection()
        try:
            results = conn.execute(sql, parameters).fetchdf()
            return results.to_dict(orient='records')
        except Exception as e:
            logger.error(f"Error listing templates: {e}")
            return []
        finally:
            conn.close()
    
    def get_hardware_platforms(self) -> List[Dict[str, Any]]:
        """
        Get all supported hardware platforms.
        
        Returns:
            List of hardware platform dictionaries
        """
        conn = self._get_connection()
        try:
            results = conn.execute(
                """
                SELECT
                    hardware_id,
                    hardware_type,
                    display_name,
                    description,
                    status
                FROM
                    hardware_platforms
                ORDER BY
                    hardware_id
                """
            ).fetchdf()
            return results.to_dict(orient='records')
        except Exception as e:
            logger.error(f"Error getting hardware platforms: {e}")
            return []
        finally:
            conn.close()
    
    def search_templates(self, query: str) -> List[Dict[str, Any]]:
        """
        Search templates by query string.
        
        Args:
            query: Search query
            
        Returns:
            List of template metadata dictionaries
        """
        search_term = f"%{query}%"
        
        conn = self._get_connection()
        try:
            results = conn.execute(
                """
                SELECT
                    model_id,
                    model_name,
                    model_family,
                    modality,
                    hardware_support,
                    last_updated
                FROM
                    model_templates
                WHERE
                    model_id LIKE ? OR
                    model_name LIKE ? OR
                    model_family LIKE ? OR
                    modality LIKE ?
                ORDER BY
                    model_family, model_id
                """,
                [search_term, search_term, search_term, search_term]
            ).fetchdf()
            return results.to_dict(orient='records')
        except Exception as e:
            logger.error(f"Error searching templates: {e}")
            return []
        finally:
            conn.close()
    
    def get_template_with_metadata(self, model_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a template with metadata by model ID.
        
        Args:
            model_id: ID of the model template to retrieve
            
        Returns:
            Template with metadata or None if not found
        """
        conn = self._get_connection()
        try:
            result = conn.execute(
                """
                SELECT
                    model_id,
                    model_name,
                    model_family,
                    modality,
                    template_content,
                    hardware_support,
                    last_updated
                FROM
                    model_templates
                WHERE
                    model_id = ?
                """,
                [model_id]
            ).fetchone()
            
            if result:
                column_names = [
                    'model_id', 'model_name', 'model_family', 'modality',
                    'template_content', 'hardware_support', 'last_updated'
                ]
                return dict(zip(column_names, result))
            else:
                logger.warning(f"Template not found for model_id: {model_id}")
                return None
        except Exception as e:
            logger.error(f"Error retrieving template with metadata: {e}")
            return None
        finally:
            conn.close()

# Legacy compatibility functions

def get_template(model_id):
    """Legacy function to get a template by model ID."""
    db = TemplateDatabase()
    return db.get_template(model_id)

def store_template(model_id, template_content):
    """Legacy function to store a template."""
    db = TemplateDatabase()
    return db.store_template(model_id, template_content)

def list_templates():
    """Legacy function to list all templates."""
    db = TemplateDatabase()
    return {t['model_id']: t for t in db.list_templates()}

# For standalone usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Template Database API")
    parser.add_argument("--list", action="store_true", help="List all templates")
    parser.add_argument("--get", metavar="MODEL_ID", help="Get a template by model ID")
    parser.add_argument("--search", metavar="QUERY", help="Search templates")
    parser.add_argument("--family", metavar="FAMILY", help="Filter by model family")
    parser.add_argument("--modality", metavar="MODALITY", help="Filter by modality")
    parser.add_argument("--hardware", action="store_true", help="List hardware platforms")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    
    args = parser.parse_args()
    
    db = TemplateDatabase(debug=args.debug)
    
    if args.list:
        templates = db.list_templates(model_family=args.family, modality=args.modality)
        print(f"Found {len(templates)} templates:")
        for template in templates:
            print(f"  - {template['model_id']} ({template['model_family']}, {template['modality']})")
    
    elif args.get:
        template = db.get_template_with_metadata(args.get)
        if template:
            print(f"Template for {template['model_id']}:")
            print(f"Model name: {template['model_name']}")
            print(f"Family: {template['model_family']}")
            print(f"Modality: {template['modality']}")
            print(f"Last updated: {template['last_updated']}")
            print("\nTemplate content:")
            print(template['template_content'][:200] + "..." if len(template['template_content']) > 200 else template['template_content'])
        else:
            print(f"Template not found for model_id: {args.get}")
    
    elif args.search:
        templates = db.search_templates(args.search)
        print(f"Found {len(templates)} matching templates:")
        for template in templates:
            print(f"  - {template['model_id']} ({template['model_family']}, {template['modality']})")
    
    elif args.hardware:
        hardware = db.get_hardware_platforms()
        print("Supported hardware platforms:")
        for hw in hardware:
            print(f"  - {hw['hardware_type']} ({hw['display_name']}): {hw['description']} [{hw['status']}]")
    
    else:
        parser.print_help()