#!/usr/bin/env python3
"""
Create template database for database-driven model template management.
This script creates or updates the template database with templates from static definitions.
"""

import os
import sys
import json
import logging
import argparse
import importlib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

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
DEFAULT_DB_PATH = "./template_db.duckdb"
DEFAULT_JSON_PATH = "./template_database.json"

# Model type definitions
MODEL_TYPES = [
    "bert", "t5", "llama", "vit", "clip", "whisper", "wav2vec2", 
    "clap", "llava", "xclip", "qwen", "detr", "default"
]

# Hardware platform definitions
HARDWARE_PLATFORMS = [
    "cpu", "cuda", "rocm", "mps", "openvino", "qualcomm", "samsung", "webnn", "webgpu"
]

# Template types
TEMPLATE_TYPES = [
    "test", "benchmark", "skill", "helper", "hardware_specific"
]

# Template definitions from static baseline
TEMPLATES = {
    "default": {
        "test": """#!/usr/bin/env python3
\"\"\"
Test for {model_name} with resource pool integration.
Generated from database template on {generated_at}
\"\"\"

import os
import unittest
import logging
from resource_pool import get_global_resource_pool

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Test{normalized_name}(unittest.TestCase):
    \"\"\"Test {model_name} with resource pool integration.\"\"\"
    
    @classmethod
    def setUpClass(cls):
        \"\"\"Set up test environment.\"\"\"
        # Get global resource pool
        cls.pool = get_global_resource_pool()
        
        # Request dependencies
        cls.torch = cls.pool.get_resource("torch", constructor=lambda: __import__("torch"))
        cls.transformers = cls.pool.get_resource("transformers", constructor=lambda: __import__("transformers"))
        
        # Check if dependencies were loaded successfully:
        if cls.torch is None or cls.transformers is None:
            raise unittest.SkipTest("Required dependencies not available")
        
        # Load model and tokenizer
        try:
            cls.tokenizer = cls.transformers.AutoTokenizer.from_pretrained("{model_name}")
            cls.model = cls.transformers.AutoModel.from_pretrained("{model_name}")
            
            # Move model to appropriate device
            cls.device = "{torch_device}"
            if cls.device != "cpu":
                cls.model = cls.model.to(cls.device)
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise unittest.SkipTest(f"Failed to load model: {e}")
    
    def test_model_loaded(self):
        \"\"\"Test that model loaded successfully.\"\"\"
        self.assertIsNotNone(self.model)
        self.assertIsNotNone(self.tokenizer)
    
    def test_inference(self):
        \"\"\"Test basic inference.\"\"\"
        # Prepare input
        text = "This is a test."
        inputs = self.tokenizer(text, return_tensors="pt")
        
        # Move inputs to device if needed:
        if self.device != "cpu":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Run inference
        with self.torch.no_grad():
            outputs = self.model(**inputs)
        
        # Verify outputs
        self.assertIsNotNone(outputs)
        self.assertIn("last_hidden_state", outputs)
        
        # Log success
        logger.info(f"Successfully tested {model_name}")

if __name__ == "__main__":
    unittest.main()
"""
    },
    "bert": {
        "test": """#!/usr/bin/env python3
\"\"\"
BERT model test for {model_name} with resource pool integration.
Generated from database template on {generated_at}
\"\"\"

import os
import unittest
import logging
from resource_pool import get_global_resource_pool

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Test{normalized_name}(unittest.TestCase):
    \"\"\"Test {model_name} with resource pool integration.\"\"\"
    
    @classmethod
    def setUpClass(cls):
        \"\"\"Set up test environment.\"\"\"
        # Get global resource pool
        cls.pool = get_global_resource_pool()
        
        # Request dependencies
        cls.torch = cls.pool.get_resource("torch", constructor=lambda: __import__("torch"))
        cls.transformers = cls.pool.get_resource("transformers", constructor=lambda: __import__("transformers"))
        
        # Check if dependencies were loaded successfully:
        if cls.torch is None or cls.transformers is None:
            raise unittest.SkipTest("Required dependencies not available")
        
        # Set up device for hardware acceleration if available
        cls.device = "cpu"
        if {has_cuda} and cls.torch.cuda.is_available():
            cls.device = "cuda"
        elif {has_mps} and hasattr(cls.torch, "mps") and cls.torch.backends.mps.is_available():
            cls.device = "mps"
        logger.info(f"Using device: {cls.device}")
        
        # Load model and tokenizer
        try:
            cls.tokenizer = cls.transformers.AutoTokenizer.from_pretrained("{model_name}")
            cls.model = cls.transformers.AutoModel.from_pretrained("{model_name}")
            
            # Move model to appropriate device
            if cls.device != "cpu":
                cls.model = cls.model.to(cls.device)
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise unittest.SkipTest(f"Failed to load model: {e}")
    
    def test_model_loaded(self):
        \"\"\"Test that model loaded successfully.\"\"\"
        self.assertIsNotNone(self.model)
        self.assertIsNotNone(self.tokenizer)
    
    def test_inference(self):
        \"\"\"Test basic inference.\"\"\"
        # Prepare input
        text = "This is a test sentence for BERT model."
        inputs = self.tokenizer(text, return_tensors="pt")
        
        # Move inputs to device if needed:
        if self.device != "cpu":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Run inference
        with self.torch.no_grad():
            outputs = self.model(**inputs)
        
        # Verify outputs
        self.assertIsNotNone(outputs)
        self.assertIn("last_hidden_state", outputs)
        
        # Check embedding dimensions
        hidden_states = outputs.last_hidden_state
        self.assertEqual(hidden_states.dim(), 3)  # [batch_size, seq_len, hidden_size]
        
        # Log success
        logger.info(f"Successfully tested {model_name}")

if __name__ == "__main__":
    unittest.main()
"""
    }
}

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Create or update template database for database-driven model template management"
    )
    parser.add_argument(
        "--db-path", type=str, default=DEFAULT_DB_PATH,
        help=f"Path to template database file (default: {DEFAULT_DB_PATH})"
    )
    parser.add_argument(
        "--json-path", type=str, default=DEFAULT_JSON_PATH,
        help=f"Path to JSON fallback file (default: {DEFAULT_JSON_PATH})"
    )
    parser.add_argument(
        "--static-dir", type=str, default="./templates",
        help="Directory with static template files to import"
    )
    parser.add_argument(
        "--create", action="store_true",
        help="Create new database (will overwrite if exists)"
    )
    parser.add_argument(
        "--update", action="store_true",
        help="Update existing database with new templates"
    )
    parser.add_argument(
        "--export", action="store_true",
        help="Export database templates to JSON file"
    )
    parser.add_argument(
        "--import", action="store_true", dest="import_json",
        help="Import templates from JSON file to database"
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List available templates in the database"
    )
    parser.add_argument(
        "--validate", action="store_true",
        help="Validate templates in the database"
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Enable debug logging"
    )
    return parser.parse_args()

def setup_environment(args):
    """Set up the environment and configure logging"""
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")

def create_database(db_path, overwrite=False):
    """Create a new template database"""
    if not DUCKDB_AVAILABLE:
        logger.error("DuckDB not available, cannot create database")
        return False
    
    # Check if database exists and handle overwrite
    db_file = Path(db_path)
    if db_file.exists() and not overwrite:
        logger.warning(f"Database file {db_path} already exists. Use --create to overwrite")
        return False
    elif db_file.exists() and overwrite:
        db_file.unlink()  # Delete the existing file
    
    logger.info(f"Creating new template database at {db_path}")
    
    try:
        # Create new database connection
        conn = duckdb.connect(db_path)
        
        # Create a simple table to store templates
        conn.execute("""
        CREATE TABLE templates (
            model_type VARCHAR,
            template_type VARCHAR,
            template TEXT,
            hardware_platform VARCHAR
        )
        """)
        
        # Populate base templates from static definitions
        for model_type, templates in TEMPLATES.items():
            for template_type, content in templates.items():
                # Insert template
                conn.execute("""
                INSERT INTO templates 
                (model_type, template_type, template, hardware_platform)
                VALUES (?, ?, ?, NULL)
                """, [model_type, template_type, content])
        
        conn.close()
        logger.info("Database created successfully")
        return True
    except Exception as e:
        logger.error(f"Error creating database: {e}")
        return False

def list_templates(db_path):
    """List templates in the database"""
    if not DUCKDB_AVAILABLE:
        logger.error("DuckDB not available, cannot list templates")
        return False
    
    try:
        conn = duckdb.connect(db_path)
        
        # Count templates
        result = conn.execute("""
        SELECT COUNT(*) FROM templates
        """).fetchone()
        template_count = result[0]
        
        # Count model types
        result = conn.execute("""
        SELECT COUNT(DISTINCT model_type) FROM templates
        """).fetchone()
        model_type_count = result[0]
        
        # Count template types
        result = conn.execute("""
        SELECT COUNT(DISTINCT template_type) FROM templates
        """).fetchone()
        template_type_count = result[0]
        
        logger.info(f"Template database contains {template_count} templates")
        logger.info(f"Covering {model_type_count} model types and {template_type_count} template types")
        
        # List templates by model type and template type
        results = conn.execute("""
        SELECT model_type, template_type, hardware_platform
        FROM templates
        ORDER BY model_type, template_type, hardware_platform
        """).fetchall()
        
        print("\nAvailable templates:")
        print("-" * 80)
        print(f"{'Model Type':<15} {'Template Type':<15} {'Hardware':<10}")
        print("-" * 80)
        
        for row in results:
            model_type, template_type, hardware = row
            hardware = hardware or "all"
            print(f"{model_type:<15} {template_type:<15} {hardware:<10}")
        
        conn.close()
        return True
    except Exception as e:
        logger.error(f"Error listing templates: {e}")
        return False

def export_to_json(db_path, json_path):
    """Export templates from database to JSON file"""
    if not DUCKDB_AVAILABLE:
        logger.error("DuckDB not available, cannot export templates")
        return False
    
    try:
        conn = duckdb.connect(db_path)
        
        # Query templates
        results = conn.execute("""
        SELECT model_type, template_type, template, hardware_platform
        FROM templates
        ORDER BY model_type, template_type, hardware_platform
        """).fetchall()
        
        # Organize templates by model type and template type
        templates_dict = {}
        for row in results:
            model_type, template_type, template, hardware = row
            
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
        conn.close()
        return True
    except Exception as e:
        logger.error(f"Error exporting templates: {e}")
        return False

def import_from_json(json_path, db_path):
    """Import templates from JSON file to database"""
    if not DUCKDB_AVAILABLE:
        logger.error("DuckDB not available, cannot import templates")
        return False
    
    try:
        # Read JSON file
        with open(json_path, 'r') as f:
            templates_dict = json.load(f)
        
        conn = duckdb.connect(db_path)
        
        # Import templates
        count = 0
        for model_type, templates in templates_dict.items():
            for key, template in templates.items():
                # Parse template type and hardware platform
                parts = key.split('_')
                template_type = parts[0]
                hardware = '_'.join(parts[1:]) if len(parts) > 1 else None
                
                conn.execute("""
                INSERT OR REPLACE INTO templates 
                (model_type, template_type, template, hardware_platform)
                VALUES (?, ?, ?, ?)
                """, [model_type, template_type, template, hardware])
                count += 1
        
        logger.info(f"Imported {count} templates from {json_path}")
        conn.close()
        return True
    except Exception as e:
        logger.error(f"Error importing templates: {e}")
        return False

def get_template_from_db(db_path, model_type, template_type, hardware_platform=None):
    """Get a template from the database"""
    if not DUCKDB_AVAILABLE:
        logger.error("DuckDB not available, cannot get template")
        return None
    
    try:
        conn = duckdb.connect(db_path)
        
        # Query for hardware-specific template first if hardware_platform provided
        if hardware_platform:
            result = conn.execute("""
            SELECT template FROM templates
            WHERE model_type = ? AND template_type = ? AND hardware_platform = ?
            """, [model_type, template_type, hardware_platform]).fetchone()
            
            if result:
                conn.close()
                return result[0]
        
        # Fall back to generic template
        result = conn.execute("""
        SELECT template FROM templates
        WHERE model_type = ? AND template_type = ? AND (hardware_platform IS NULL OR hardware_platform = '')
        """, [model_type, template_type]).fetchone()
        
        if result:
            conn.close()
            return result[0]
        
        # Fall back to default template type
        result = conn.execute("""
        SELECT template FROM templates
        WHERE model_type = 'default' AND template_type = ? AND (hardware_platform IS NULL OR hardware_platform = '')
        """, [template_type]).fetchone()
        
        conn.close()
        return result[0] if result else None
    except Exception as e:
        logger.error(f"Error getting template: {e}")
        return None

def main():
    """Main function"""
    args = parse_args()
    setup_environment(args)
    
    # Check for DuckDB availability
    if not DUCKDB_AVAILABLE and not args.export:
        logger.warning("DuckDB not available, using JSON fallback")
    
    # Create new database if requested
    if args.create:
        if not create_database(args.db_path, overwrite=True):
            return 1
    
    # List templates if requested
    if args.list:
        if not list_templates(args.db_path):
            return 1
    
    # Export templates to JSON if requested
    if args.export:
        if not export_to_json(args.db_path, args.json_path):
            return 1
    
    # Import templates from JSON if requested
    if args.import_json:
        if not import_from_json(args.json_path, args.db_path):
            return 1
    
    # Check if any action was performed
    if not any([args.create, args.list, args.export, args.import_json]):
        logger.error("No action specified. Use --create, --list, --export, or --import")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())