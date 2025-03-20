#!/usr/bin/env python3
"""
Distributed Testing Framework - Test Generator Integration

This module provides integration between template-based test generators and the
distributed testing framework. It enables dynamic generation of tests based on
templates stored in the DuckDB database, with support for model-specific and
hardware-specific customization.

Core features:
- Dynamic test generation from templates in DuckDB
- Model family detection and template selection
- Hardware-specific template customization
- Task dependency management for proper execution order
- Efficient batch submission to the coordinator
"""

import os
import sys
import json
import time
import uuid
import logging
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s'
)
logger = logging.getLogger("test_generator_integration")

# Add parent directory to path to import modules from parent
parent_dir = str(Path(__file__).parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Conditional imports
try:
    import duckdb
    DUCKDB_AVAILABLE = True
except ImportError:
    logger.warning("DuckDB not available. Integration will use fallback mechanisms.")
    DUCKDB_AVAILABLE = False


class TestGeneratorIntegration:
    """Integrates with template-based test generators for dynamic test creation."""
    
    def __init__(self, template_db_path, coordinator_url=None, coordinator_client=None):
        """Initialize with template database path and coordinator URL.
        
        Args:
            template_db_path: Path to the template database (DuckDB)
            coordinator_url: URL of the coordinator (if coordinator_client not provided)
            coordinator_client: Instance of CoordinatorClient (optional)
        """
        self.template_db_path = template_db_path
        self.coordinator_url = coordinator_url
        
        # Initialize template database connection
        if DUCKDB_AVAILABLE:
            try:
                self.template_db = duckdb.connect(template_db_path)
                self._ensure_schema_exists()
                logger.info(f"Connected to template database at {template_db_path}")
            except Exception as e:
                logger.error(f"Error connecting to template database: {e}")
                self.template_db = None
        else:
            self.template_db = None
            logger.warning("DuckDB not available, template database functionality limited")
        
        # Set up coordinator client
        self.coordinator_client = coordinator_client
        if not coordinator_client and coordinator_url:
            # Import here to avoid circular imports
            try:
                from coordinator_client import CoordinatorClient
                self.coordinator_client = CoordinatorClient(coordinator_url)
                logger.info(f"Created coordinator client for {coordinator_url}")
            except ImportError:
                logger.warning("CoordinatorClient not available, task submission disabled")
                self.coordinator_client = None
    
    def _ensure_schema_exists(self):
        """Create template database schema if it doesn't exist."""
        if not self.template_db:
            return
            
        try:
            # Check if templates table exists
            table_exists = self.template_db.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name='templates'
            """).fetchone() is not None
            
            if not table_exists:
                logger.info("Creating templates table")
                
                self.template_db.execute("""
                    CREATE TABLE templates (
                        template_id INTEGER PRIMARY KEY,
                        template_name VARCHAR,
                        model_family VARCHAR,
                        hardware_type VARCHAR,
                        content TEXT,
                        description VARCHAR,
                        created_at VARCHAR,
                        updated_at VARCHAR,
                        version INTEGER
                    )
                """)
                
                # Create indices for faster querying
                self.template_db.execute("CREATE INDEX idx_templates_model_family ON templates(model_family)")
                self.template_db.execute("CREATE INDEX idx_templates_hardware_type ON templates(hardware_type)")
                
                logger.info("Created templates table and indices")
            
            # Check if model_mapping table exists
            mapping_exists = self.template_db.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name='model_mapping'
            """).fetchone() is not None
            
            if not mapping_exists:
                logger.info("Creating model_mapping table")
                
                self.template_db.execute("""
                    CREATE TABLE model_mapping (
                        mapping_id INTEGER PRIMARY KEY,
                        model_name VARCHAR UNIQUE,
                        model_family VARCHAR,
                        template_id INTEGER,
                        description VARCHAR,
                        FOREIGN KEY (template_id) REFERENCES templates(template_id)
                    )
                """)
                
                self.template_db.execute("CREATE INDEX idx_model_mapping_model_name ON model_mapping(model_name)")
                
                logger.info("Created model_mapping table and indices")
                
        except Exception as e:
            logger.error(f"Error ensuring schema exists: {e}")
            raise
    
    def generate_and_submit_tests(self, model_name, hardware_types=None, batch_sizes=None):
        """Generate tests from templates and submit to the coordinator.
        
        Args:
            model_name: Name of the model to generate tests for
            hardware_types: List of hardware types to target (if None, use all available)
            batch_sizes: List of batch sizes to test (if None, use [1, 4])
            
        Returns:
            Tuple[bool, List]: (success, list of generated test definitions)
        """
        # Determine model family
        model_family = self._get_model_family(model_name)
        
        # Find appropriate templates
        templates = self._fetch_templates(model_family, hardware_types)
        
        if not templates:
            logger.warning(f"No templates found for model {model_name} ({model_family})")
            return False, []
        
        # Generate tests
        generated_tests = []
        for template in templates:
            for hardware in hardware_types or ["cpu"]:
                for batch_size in batch_sizes or [1, 4]:
                    test_config = {
                        "model_name": model_name,
                        "model_family": model_family,
                        "hardware_type": hardware,
                        "batch_size": batch_size,
                        "template_id": template["template_id"]
                    }
                    test = self._generate_test(template, test_config)
                    generated_tests.append(test)
        
        # Set up dependencies between tests
        tests_with_dependencies = self._setup_dependencies(generated_tests)
        
        # Submit to coordinator if available
        if self.coordinator_client:
            submission_results = []
            for test in tests_with_dependencies:
                result = self.coordinator_client.submit_task(test)
                submission_results.append(result)
            
            success = all(result.get("success", False) for result in submission_results)
            logger.info(f"Submitted {len(tests_with_dependencies)} tests, success: {success}")
            return success, tests_with_dependencies
        else:
            logger.info(f"Generated {len(tests_with_dependencies)} tests (coordinator not available, not submitted)")
            return True, tests_with_dependencies
    
    def _get_model_family(self, model_name):
        """Determine the model family from the model name.
        
        Args:
            model_name: Name of the model
            
        Returns:
            str: Model family name
        """
        if not self.template_db:
            # Fallback to inference
            return self._infer_model_family(model_name)
            
        # Query the template database to find model family
        try:
            result = self.template_db.execute(f"""
                SELECT model_family FROM model_mapping 
                WHERE model_name = ?
            """, [model_name]).fetchone()
            
            if result:
                return result[0]
        except Exception as e:
            logger.warning(f"Error querying model family: {e}")
            
        # Try to infer from name if not found
        return self._infer_model_family(model_name)
    
    def _infer_model_family(self, model_name):
        """Infer the model family from the model name.
        
        Args:
            model_name: Name of the model
            
        Returns:
            str: Inferred model family name
        """
        model_name_lower = model_name.lower()
        
        # Text embedding models
        if "bert" in model_name_lower or "roberta" in model_name_lower:
            return "text_embedding"
            
        # Text generation models
        if "t5" in model_name_lower or "gpt" in model_name_lower or "llama" in model_name_lower:
            return "text_generation"
            
        # Vision models
        if "vit" in model_name_lower or "resnet" in model_name_lower or "efficientnet" in model_name_lower:
            return "vision"
            
        # Audio models
        if "whisper" in model_name_lower or "wav2vec" in model_name_lower:
            return "audio"
            
        # Multimodal models
        if "clip" in model_name_lower or "llava" in model_name_lower:
            return "multimodal"
            
        return "unknown"
    
    def _fetch_templates(self, model_family, hardware_types=None):
        """Fetch appropriate templates for the model family and hardware types.
        
        Args:
            model_family: Model family to fetch templates for
            hardware_types: List of hardware types to fetch templates for
            
        Returns:
            List[Dict]: List of template dictionaries
        """
        if not self.template_db:
            logger.warning("Template database not available, returning empty templates list")
            return []
            
        try:
            query = f"""
                SELECT * FROM templates 
                WHERE model_family = ?
            """
            params = [model_family]
            
            if hardware_types:
                placeholders = ", ".join(["?" for _ in hardware_types])
                query += f" AND (hardware_type IN ({placeholders}) OR hardware_type IS NULL)"
                params.extend(hardware_types)
                
            results = self.template_db.execute(query, params).fetchall()
            
            # Convert to dictionaries
            templates = []
            for row in results:
                template = {
                    "template_id": row[0],
                    "template_name": row[1],
                    "model_family": row[2],
                    "hardware_type": row[3],
                    "content": row[4],
                    "description": row[5],
                    "created_at": row[6],
                    "updated_at": row[7],
                    "version": row[8]
                }
                templates.append(template)
                
            return templates
        except Exception as e:
            logger.error(f"Error fetching templates: {e}")
            return []
    
    def _generate_test(self, template, config):
        """Generate a test from a template and configuration.
        
        Args:
            template: Template dictionary
            config: Configuration dictionary
            
        Returns:
            Dict: Test definition
        """
        # Extract template content
        template_content = template["content"]
        
        # Perform variable substitution
        for key, value in config.items():
            template_content = template_content.replace(f"${{{key}}}", str(value))
        
        # Create task definition
        task = {
            "test_id": str(uuid.uuid4()),
            "model_name": config["model_name"],
            "model_family": config["model_family"],
            "hardware_type": config["hardware_type"],
            "batch_size": config["batch_size"],
            "test_content": template_content,
            "test_type": "generated",
            "priority": self._calculate_priority(config),
            "requirements": {
                "hardware_type": config["hardware_type"],
                "min_memory_gb": self._estimate_memory_requirement(config),
                "test_timeout_seconds": 600  # 10 minutes default timeout
            }
        }
        
        return task
    
    def _setup_dependencies(self, tests):
        """Set up dependencies between tests for proper execution order.
        
        Args:
            tests: List of test definitions
            
        Returns:
            List[Dict]: Tests with dependencies
        """
        # Group by model and hardware
        tests_by_group = {}
        for test in tests:
            key = (test["model_name"], test["hardware_type"])
            if key not in tests_by_group:
                tests_by_group[key] = []
            tests_by_group[key].append(test)
        
        # Set up dependencies within each group
        result = []
        for group_tests in tests_by_group.values():
            # Sort by batch size (ascending)
            sorted_tests = sorted(group_tests, key=lambda t: t["batch_size"])
            
            # Set up dependencies (each test depends on the previous one)
            for i in range(1, len(sorted_tests)):
                if "dependencies" not in sorted_tests[i]:
                    sorted_tests[i]["dependencies"] = []
                sorted_tests[i]["dependencies"].append(sorted_tests[i-1]["test_id"])
            
            result.extend(sorted_tests)
        
        return result
    
    def _calculate_priority(self, config):
        """Calculate task priority based on configuration.
        
        Args:
            config: Test configuration
            
        Returns:
            int: Priority value (1-10, lower is higher priority)
        """
        base_priority = 5  # Default priority
        
        # Adjust based on hardware type
        if config["hardware_type"] == "cpu":
            base_priority += 1  # Slightly higher priority for CPU tests
        elif config["hardware_type"] in ["cuda", "rocm"]:
            base_priority -= 1  # Slightly lower for GPU tests (more resource intensive)
        
        # Adjust based on batch size
        if config["batch_size"] <= 1:
            base_priority += 1  # Higher priority for small batch sizes
        elif config["batch_size"] >= 16:
            base_priority -= 1  # Lower priority for large batch sizes
            
        return max(1, min(10, base_priority))  # Ensure between 1-10
    
    def _estimate_memory_requirement(self, config):
        """Estimate memory requirement based on configuration.
        
        Args:
            config: Test configuration
            
        Returns:
            float: Estimated memory requirement in GB
        """
        base_memory = 1.0  # Default 1GB
        
        # Adjust based on model family
        if config["model_family"] == "text_generation":
            base_memory = 4.0
        elif config["model_family"] == "vision":
            base_memory = 2.0
        elif config["model_family"] == "multimodal":
            base_memory = 6.0
        elif config["model_family"] == "audio":
            base_memory = 3.0
            
        # Adjust based on batch size
        batch_factor = config["batch_size"] / 4.0  # Normalized to batch size 4
        memory_estimate = base_memory * batch_factor
        
        return max(0.5, memory_estimate)  # Minimum 0.5GB
    
    def add_template(self, template_name, model_family, content, hardware_type=None, description=None):
        """Add a new template to the database.
        
        Args:
            template_name: Name of the template
            model_family: Model family the template applies to
            content: Template content (Python code with variable placeholders)
            hardware_type: Hardware type the template is optimized for (optional)
            description: Description of the template
            
        Returns:
            int: Template ID if successful, None otherwise
        """
        if not self.template_db:
            logger.error("Template database not available, cannot add template")
            return None
            
        try:
            # Insert template
            created_at = datetime.now().isoformat()
            self.template_db.execute("""
                INSERT INTO templates (
                    template_name, model_family, hardware_type, content, 
                    description, created_at, updated_at, version
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                template_name, model_family, hardware_type, content,
                description or f"Template for {model_family} models",
                created_at, created_at, 1
            ])
            
            # Get the inserted template ID
            result = self.template_db.execute("""
                SELECT template_id FROM templates
                WHERE template_name = ? AND model_family = ? AND created_at = ?
            """, [template_name, model_family, created_at]).fetchone()
            
            if result:
                logger.info(f"Added template {template_name} with ID {result[0]}")
                return result[0]
                
            return None
        except Exception as e:
            logger.error(f"Error adding template: {e}")
            return None
    
    def add_model_mapping(self, model_name, model_family, template_id=None, description=None):
        """Add a model-to-family mapping to the database.
        
        Args:
            model_name: Name of the model
            model_family: Model family
            template_id: ID of a specific template to use (optional)
            description: Description of the mapping
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.template_db:
            logger.error("Template database not available, cannot add mapping")
            return False
            
        try:
            # Check if mapping already exists
            existing = self.template_db.execute("""
                SELECT mapping_id FROM model_mapping
                WHERE model_name = ?
            """, [model_name]).fetchone()
            
            if existing:
                # Update existing mapping
                self.template_db.execute("""
                    UPDATE model_mapping
                    SET model_family = ?, template_id = ?, description = ?
                    WHERE model_name = ?
                """, [model_family, template_id, description, model_name])
                logger.info(f"Updated mapping for {model_name} to {model_family}")
            else:
                # Insert new mapping
                self.template_db.execute("""
                    INSERT INTO model_mapping (
                        model_name, model_family, template_id, description
                    ) VALUES (?, ?, ?, ?)
                """, [model_name, model_family, template_id, description])
                logger.info(f"Added mapping for {model_name} to {model_family}")
                
            return True
        except Exception as e:
            logger.error(f"Error adding model mapping: {e}")
            return False
    
    def get_template_by_id(self, template_id):
        """Get a template by ID.
        
        Args:
            template_id: ID of the template to retrieve
            
        Returns:
            Dict: Template dictionary or None if not found
        """
        if not self.template_db:
            logger.error("Template database not available, cannot get template")
            return None
            
        try:
            result = self.template_db.execute("""
                SELECT * FROM templates WHERE template_id = ?
            """, [template_id]).fetchone()
            
            if result:
                template = {
                    "template_id": result[0],
                    "template_name": result[1],
                    "model_family": result[2],
                    "hardware_type": result[3],
                    "content": result[4],
                    "description": result[5],
                    "created_at": result[6],
                    "updated_at": result[7],
                    "version": result[8]
                }
                return template
                
            return None
        except Exception as e:
            logger.error(f"Error getting template: {e}")
            return None
    
    def get_mapping_by_model(self, model_name):
        """Get model mapping for a specific model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Dict: Mapping dictionary or None if not found
        """
        if not self.template_db:
            logger.error("Template database not available, cannot get mapping")
            return None
            
        try:
            result = self.template_db.execute("""
                SELECT * FROM model_mapping WHERE model_name = ?
            """, [model_name]).fetchone()
            
            if result:
                mapping = {
                    "mapping_id": result[0],
                    "model_name": result[1],
                    "model_family": result[2],
                    "template_id": result[3],
                    "description": result[4]
                }
                return mapping
                
            return None
        except Exception as e:
            logger.error(f"Error getting model mapping: {e}")
            return None
    
    def close(self):
        """Close database connections and free resources."""
        if self.template_db:
            self.template_db.close()
            logger.info("Closed template database connection")


# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Generator Integration")
    parser.add_argument("--template-db", required=True, help="Path to template database")
    parser.add_argument("--add-template", action="store_true", help="Add a template")
    parser.add_argument("--add-mapping", action="store_true", help="Add a model mapping")
    parser.add_argument("--generate", action="store_true", help="Generate tests")
    parser.add_argument("--list-templates", action="store_true", help="List templates")
    parser.add_argument("--model", help="Model name for generation or mapping")
    parser.add_argument("--hardware", help="Hardware type(s), comma-separated")
    parser.add_argument("--batch-sizes", help="Batch size(s), comma-separated")
    parser.add_argument("--template-name", help="Template name for adding")
    parser.add_argument("--model-family", help="Model family for adding")
    parser.add_argument("--template-file", help="File containing template content")
    parser.add_argument("--description", help="Description for template or mapping")
    
    args = parser.parse_args()
    
    # Initialize integration
    integration = TestGeneratorIntegration(args.template_db)
    
    try:
        if args.add_template:
            if not args.template_name or not args.model_family or not args.template_file:
                logger.error("Template name, model family, and template file are required for adding a template")
                sys.exit(1)
                
            with open(args.template_file, 'r') as f:
                content = f.read()
                
            template_id = integration.add_template(
                args.template_name,
                args.model_family,
                content,
                args.hardware,
                args.description
            )
            
            if template_id:
                print(f"Added template with ID {template_id}")
            else:
                print("Failed to add template")
                
        elif args.add_mapping:
            if not args.model or not args.model_family:
                logger.error("Model name and model family are required for adding a mapping")
                sys.exit(1)
                
            success = integration.add_model_mapping(
                args.model,
                args.model_family,
                description=args.description
            )
            
            if success:
                print(f"Added mapping for {args.model} to {args.model_family}")
            else:
                print("Failed to add mapping")
                
        elif args.generate:
            if not args.model:
                logger.error("Model name is required for test generation")
                sys.exit(1)
                
            hardware_types = args.hardware.split(',') if args.hardware else None
            batch_sizes = [int(b) for b in args.batch_sizes.split(',')] if args.batch_sizes else None
            
            success, tests = integration.generate_and_submit_tests(
                args.model,
                hardware_types,
                batch_sizes
            )
            
            if success:
                print(f"Generated {len(tests)} tests")
                for i, test in enumerate(tests):
                    print(f"Test {i+1}: {test['model_name']} on {test['hardware_type']} with batch size {test['batch_size']}")
            else:
                print("Failed to generate tests")
                
        elif args.list_templates:
            if not integration.template_db:
                print("Template database not available")
                sys.exit(1)
                
            templates = integration.template_db.execute("SELECT * FROM templates").fetchall()
            
            if templates:
                print(f"Found {len(templates)} templates:")
                for template in templates:
                    print(f"ID: {template[0]}, Name: {template[1]}, Family: {template[2]}, Hardware: {template[3]}")
            else:
                print("No templates found")
        else:
            print("No action specified. Use --add-template, --add-mapping, --generate, or --list-templates")
    finally:
        integration.close()