#!/usr/bin/env python3
"""
Template Database for End-to-End Testing Framework

This module provides a DuckDB-based database for storing and retrieving templates for:
1. Model skill implementations
2. Test implementations
3. Benchmark implementations
4. Documentation templates

The database enables efficient template management with:
- Template versioning and inheritance
- Hardware-specific template variations
- Model family mappings
- Variable substitution capabilities
"""

import os
import re
import json
import uuid
import logging
import datetime
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any, Union

# Check if DuckDB is available
try:
    import duckdb
    HAS_DUCKDB = True
except ImportError:
    HAS_DUCKDB = False
    
# Setup logging
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Constants
DEFAULT_DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "template_database.duckdb")
MODEL_FAMILIES = [
    "text_embedding", "text_generation", "vision", "audio", "multimodal", 
    "audio_classification", "vision_classification", "object_detection"
]
HARDWARE_PLATFORMS = [
    "cpu", "cuda", "rocm", "mps", "openvino", "qnn", "webnn", "webgpu"
]
TEMPLATE_TYPES = ["skill", "test", "benchmark", "documentation", "helper"]

class TemplateDatabase:
    """
    Database for storing and retrieving test templates.
    
    This class provides a DuckDB-based storage system for templates that can be 
    used to generate tests, skills, benchmarks, and documentation for different
    model families and hardware platforms.
    """
    
    def __init__(self, db_path: str = DEFAULT_DB_PATH, verbose: bool = False):
        """
        Initialize the template database.
        
        Args:
            db_path: Path to the DuckDB database
            verbose: Enable verbose logging
        """
        if not HAS_DUCKDB:
            raise ImportError("DuckDB is required for template database functionality. Install with: pip install duckdb")
            
        self.db_path = db_path
        self.verbose = verbose
        
        if verbose:
            logger.setLevel(logging.DEBUG)
            
        # Initialize the database if it doesn't exist
        self._initialize_db()
        
    def _initialize_db(self):
        """Initialize the database schema if it doesn't exist."""
        logger.debug(f"Initializing template database at {self.db_path}")
        
        # Connect to the database
        conn = duckdb.connect(self.db_path)
        
        try:
            # Create tables if they don't exist
            
            # Templates table - stores the actual templates
            conn.execute("""
                CREATE TABLE IF NOT EXISTS templates (
                    template_id VARCHAR PRIMARY KEY,
                    template_name VARCHAR NOT NULL,
                    template_type VARCHAR NOT NULL,
                    model_family VARCHAR NOT NULL,
                    hardware_platform VARCHAR,
                    template_content TEXT NOT NULL,
                    description TEXT,
                    version VARCHAR NOT NULL,
                    parent_template_id VARCHAR,
                    created_at TIMESTAMP NOT NULL,
                    updated_at TIMESTAMP NOT NULL,
                    is_active BOOLEAN NOT NULL
                )
            """)
            
            # Template variables table - defines variables used in templates
            conn.execute("""
                CREATE TABLE IF NOT EXISTS template_variables (
                    variable_id VARCHAR PRIMARY KEY,
                    template_id VARCHAR NOT NULL,
                    variable_name VARCHAR NOT NULL,
                    variable_description TEXT,
                    default_value VARCHAR,
                    is_required BOOLEAN NOT NULL,
                    created_at TIMESTAMP NOT NULL,
                    FOREIGN KEY (template_id) REFERENCES templates(template_id)
                )
            """)
            
            # Model mappings table - maps models to model families
            conn.execute("""
                CREATE TABLE IF NOT EXISTS model_mappings (
                    mapping_id VARCHAR PRIMARY KEY,
                    model_name VARCHAR UNIQUE NOT NULL,
                    model_family VARCHAR NOT NULL,
                    description TEXT,
                    created_at TIMESTAMP NOT NULL,
                    updated_at TIMESTAMP NOT NULL
                )
            """)
            
            # Template dependency table - tracks dependencies between templates
            conn.execute("""
                CREATE TABLE IF NOT EXISTS template_dependencies (
                    dependency_id VARCHAR PRIMARY KEY,
                    template_id VARCHAR NOT NULL,
                    depends_on_template_id VARCHAR NOT NULL,
                    dependency_type VARCHAR NOT NULL,
                    created_at TIMESTAMP NOT NULL,
                    FOREIGN KEY (template_id) REFERENCES templates(template_id),
                    FOREIGN KEY (depends_on_template_id) REFERENCES templates(template_id),
                    UNIQUE (template_id, depends_on_template_id)
                )
            """)
            
            # Hardware compatibility table - defines hardware compatibility for models
            conn.execute("""
                CREATE TABLE IF NOT EXISTS hardware_compatibility (
                    compatibility_id VARCHAR PRIMARY KEY,
                    model_family VARCHAR NOT NULL,
                    hardware_platform VARCHAR NOT NULL,
                    compatibility_level VARCHAR NOT NULL,
                    description TEXT,
                    created_at TIMESTAMP NOT NULL,
                    updated_at TIMESTAMP NOT NULL,
                    UNIQUE (model_family, hardware_platform)
                )
            """)
            
            # Create indexes for performance
            conn.execute("CREATE INDEX IF NOT EXISTS idx_templates_model_family ON templates(model_family)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_templates_hardware ON templates(hardware_platform)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_templates_type ON templates(template_type)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_model_mappings_model ON model_mappings(model_name)")
            
            logger.debug("Template database schema initialized successfully")
            
        finally:
            conn.close()
            
    def add_template(self, 
                     template_name: str,
                     template_type: str,
                     model_family: str,
                     template_content: str,
                     hardware_platform: Optional[str] = None,
                     description: Optional[str] = None,
                     version: str = "1.0.0",
                     parent_template_id: Optional[str] = None) -> str:
        """
        Add a new template to the database.
        
        Args:
            template_name: Name of the template
            template_type: Type of template (skill, test, benchmark, documentation, helper)
            model_family: Model family (text_embedding, vision, etc.)
            template_content: Template content with variables
            hardware_platform: Specific hardware platform (optional)
            description: Template description (optional)
            version: Template version (default: "1.0.0")
            parent_template_id: ID of parent template for inheritance (optional)
            
        Returns:
            Template ID
        """
        # Validate inputs
        if template_type not in TEMPLATE_TYPES:
            raise ValueError(f"Invalid template type: {template_type}. Must be one of: {TEMPLATE_TYPES}")
            
        if model_family not in MODEL_FAMILIES:
            raise ValueError(f"Invalid model family: {model_family}. Must be one of: {MODEL_FAMILIES}")
            
        if hardware_platform and hardware_platform not in HARDWARE_PLATFORMS:
            raise ValueError(f"Invalid hardware platform: {hardware_platform}. Must be one of: {HARDWARE_PLATFORMS}")
            
        # Extract variables from the template
        variables = self._extract_variables(template_content)
        
        # Generate a unique ID
        template_id = str(uuid.uuid4())
        now = datetime.datetime.now()
        
        # Connect to the database
        conn = duckdb.connect(self.db_path)
        
        try:
            # Insert the template
            conn.execute("""
                INSERT INTO templates 
                (template_id, template_name, template_type, model_family, hardware_platform, 
                 template_content, description, version, parent_template_id, created_at, updated_at, is_active)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (template_id, template_name, template_type, model_family, hardware_platform, 
                template_content, description, version, parent_template_id, now, now, True))
            
            # Insert variables
            for var_name, var_info in variables.items():
                var_id = str(uuid.uuid4())
                conn.execute("""
                    INSERT INTO template_variables
                    (variable_id, template_id, variable_name, variable_description, default_value, is_required, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (var_id, template_id, var_name, var_info.get("description"), 
                     var_info.get("default"), var_info.get("required", True), now))
            
            logger.info(f"Added template: {template_name} ({template_id}) for {model_family}")
            return template_id
            
        finally:
            conn.close()
    
    def get_template(self, 
                     model_family: str,
                     template_type: str,
                     hardware_platform: Optional[str] = None,
                     template_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get a template from the database.
        
        Args:
            model_family: Model family (text_embedding, vision, etc.)
            template_type: Type of template (skill, test, benchmark, documentation, helper)
            hardware_platform: Specific hardware platform (optional)
            template_name: Specific template name (optional)
            
        Returns:
            Template dictionary including content and variables
        """
        # Connect to the database
        conn = duckdb.connect(self.db_path)
        
        try:
            # Build query based on provided parameters
            query = """
                SELECT * FROM templates 
                WHERE model_family = ? AND template_type = ? AND is_active = TRUE
            """
            params = [model_family, template_type]
            
            if hardware_platform:
                query += " AND (hardware_platform = ? OR hardware_platform IS NULL)"
                params.append(hardware_platform)
            else:
                query += " AND hardware_platform IS NULL"
                
            if template_name:
                query += " AND template_name = ?"
                params.append(template_name)
                
            # Order by hardware specificity (hardware-specific templates first)
            # and then by creation date (newest first)
            query += " ORDER BY CASE WHEN hardware_platform IS NULL THEN 0 ELSE 1 END DESC, created_at DESC LIMIT 1"
            
            # Execute the query
            result = conn.execute(query, params).fetchone()
            
            if not result:
                logger.warning(f"No template found for model_family={model_family}, template_type={template_type}, hardware={hardware_platform}")
                return None
                
            # Convert to dictionary
            template = {
                "template_id": result[0],
                "template_name": result[1],
                "template_type": result[2],
                "model_family": result[3],
                "hardware_platform": result[4],
                "template_content": result[5],
                "description": result[6],
                "version": result[7],
                "parent_template_id": result[8],
                "created_at": result[9],
                "updated_at": result[10],
                "is_active": result[11]
            }
            
            # Get variables for this template
            variables = conn.execute("""
                SELECT variable_name, variable_description, default_value, is_required
                FROM template_variables
                WHERE template_id = ?
            """, [template["template_id"]]).fetchall()
            
            template["variables"] = {}
            for var in variables:
                template["variables"][var[0]] = {
                    "description": var[1],
                    "default": var[2],
                    "required": var[3]
                }
                
            # Get dependencies for this template
            dependencies = conn.execute("""
                SELECT t.template_id, t.template_name, t.template_type, td.dependency_type
                FROM template_dependencies td
                JOIN templates t ON td.depends_on_template_id = t.template_id
                WHERE td.template_id = ?
            """, [template["template_id"]]).fetchall()
            
            template["dependencies"] = []
            for dep in dependencies:
                template["dependencies"].append({
                    "template_id": dep[0],
                    "template_name": dep[1],
                    "template_type": dep[2],
                    "dependency_type": dep[3]
                })
                
            logger.debug(f"Retrieved template: {template['template_name']} ({template['template_id']})")
            return template
            
        finally:
            conn.close()
            
    def add_model_mapping(self, model_name: str, model_family: str, description: Optional[str] = None) -> str:
        """
        Add a mapping from model name to model family.
        
        Args:
            model_name: Name of the model (e.g., "bert-base-uncased")
            model_family: Model family (e.g., "text_embedding")
            description: Optional description
            
        Returns:
            Mapping ID
        """
        # Validate model family
        if model_family not in MODEL_FAMILIES:
            raise ValueError(f"Invalid model family: {model_family}. Must be one of: {MODEL_FAMILIES}")
            
        # Generate a unique ID
        mapping_id = str(uuid.uuid4())
        now = datetime.datetime.now()
        
        # Connect to the database
        conn = duckdb.connect(self.db_path)
        
        try:
            # Check if mapping already exists
            exists = conn.execute("""
                SELECT COUNT(*) FROM model_mappings WHERE model_name = ?
            """, [model_name]).fetchone()[0]
            
            if exists:
                # Update existing mapping
                conn.execute("""
                    UPDATE model_mappings
                    SET model_family = ?, description = ?, updated_at = ?
                    WHERE model_name = ?
                """, (model_family, description, now, model_name))
                
                # Get the existing ID
                mapping_id = conn.execute("""
                    SELECT mapping_id FROM model_mappings WHERE model_name = ?
                """, [model_name]).fetchone()[0]
                
                logger.info(f"Updated model mapping: {model_name} -> {model_family}")
            else:
                # Insert new mapping
                conn.execute("""
                    INSERT INTO model_mappings
                    (mapping_id, model_name, model_family, description, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (mapping_id, model_name, model_family, description, now, now))
                
                logger.info(f"Added model mapping: {model_name} -> {model_family}")
                
            return mapping_id
            
        finally:
            conn.close()
            
    def get_model_family(self, model_name: str) -> Optional[str]:
        """
        Get the model family for a given model name.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Model family or None if not found
        """
        # Connect to the database
        conn = duckdb.connect(self.db_path)
        
        try:
            # Query the model mapping
            result = conn.execute("""
                SELECT model_family FROM model_mappings WHERE model_name = ?
            """, [model_name]).fetchone()
            
            if result:
                return result[0]
                
            # If not found, try to infer from model name
            model_family = self._infer_model_family(model_name)
            if model_family:
                logger.info(f"Inferred model family for {model_name}: {model_family}")
                return model_family
                
            logger.warning(f"Could not determine model family for: {model_name}")
            return None
            
        finally:
            conn.close()
            
    def _infer_model_family(self, model_name: str) -> Optional[str]:
        """
        Infer the model family from the model name using heuristics.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Inferred model family or None
        """
        model_name = model_name.lower()
        
        # Text embedding models
        if any(x in model_name for x in ["bert", "roberta", "distilbert", "sentence", "bge", "all-mpnet", "e5-"]):
            return "text_embedding"
            
        # Text generation models
        if any(x in model_name for x in ["gpt", "llama", "t5", "bart", "palm", "bloom", "mixtral", "qwen", "mistral"]):
            return "text_generation"
            
        # Vision models
        if any(x in model_name for x in ["vit", "resnet", "efficientnet", "convnext", "swin"]):
            return "vision"
            
        # Audio models
        if any(x in model_name for x in ["whisper", "wav2vec", "hubert", "clap"]):
            return "audio"
            
        # Multimodal models
        if any(x in model_name for x in ["clip", "blip", "llava", "pix2struct"]):
            return "multimodal"
            
        # Object detection
        if any(x in model_name for x in ["yolo", "detr", "faster-rcnn", "maskrcnn"]):
            return "object_detection"
            
        return None
        
    def add_hardware_compatibility(self, 
                                  model_family: str,
                                  hardware_platform: str,
                                  compatibility_level: str,
                                  description: Optional[str] = None) -> str:
        """
        Add hardware compatibility information for a model family.
        
        Args:
            model_family: Model family
            hardware_platform: Hardware platform
            compatibility_level: Compatibility level (full, limited, none)
            description: Optional description
            
        Returns:
            Compatibility ID
        """
        # Validate inputs
        if model_family not in MODEL_FAMILIES:
            raise ValueError(f"Invalid model family: {model_family}. Must be one of: {MODEL_FAMILIES}")
            
        if hardware_platform not in HARDWARE_PLATFORMS:
            raise ValueError(f"Invalid hardware platform: {hardware_platform}. Must be one of: {HARDWARE_PLATFORMS}")
            
        if compatibility_level not in ["full", "limited", "none"]:
            raise ValueError(f"Invalid compatibility level: {compatibility_level}. Must be one of: full, limited, none")
            
        # Generate a unique ID
        compatibility_id = str(uuid.uuid4())
        now = datetime.datetime.now()
        
        # Connect to the database
        conn = duckdb.connect(self.db_path)
        
        try:
            # Check if compatibility entry already exists
            exists = conn.execute("""
                SELECT COUNT(*) FROM hardware_compatibility 
                WHERE model_family = ? AND hardware_platform = ?
            """, [model_family, hardware_platform]).fetchone()[0]
            
            if exists:
                # Update existing entry
                conn.execute("""
                    UPDATE hardware_compatibility
                    SET compatibility_level = ?, description = ?, updated_at = ?
                    WHERE model_family = ? AND hardware_platform = ?
                """, (compatibility_level, description, now, model_family, hardware_platform))
                
                # Get the existing ID
                compatibility_id = conn.execute("""
                    SELECT compatibility_id FROM hardware_compatibility 
                    WHERE model_family = ? AND hardware_platform = ?
                """, [model_family, hardware_platform]).fetchone()[0]
                
                logger.info(f"Updated hardware compatibility: {model_family} on {hardware_platform} -> {compatibility_level}")
            else:
                # Insert new entry
                conn.execute("""
                    INSERT INTO hardware_compatibility
                    (compatibility_id, model_family, hardware_platform, compatibility_level, description, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (compatibility_id, model_family, hardware_platform, compatibility_level, description, now, now))
                
                logger.info(f"Added hardware compatibility: {model_family} on {hardware_platform} -> {compatibility_level}")
                
            return compatibility_id
            
        finally:
            conn.close()
            
    def get_hardware_compatibility(self, model_family: str, hardware_platform: str) -> Dict[str, Any]:
        """
        Get hardware compatibility information.
        
        Args:
            model_family: Model family
            hardware_platform: Hardware platform
            
        Returns:
            Compatibility information dict
        """
        # Connect to the database
        conn = duckdb.connect(self.db_path)
        
        try:
            # Query the hardware compatibility
            result = conn.execute("""
                SELECT compatibility_level, description FROM hardware_compatibility 
                WHERE model_family = ? AND hardware_platform = ?
            """, [model_family, hardware_platform]).fetchone()
            
            if result:
                return {
                    "compatibility_level": result[0],
                    "description": result[1]
                }
                
            # Default to limited compatibility if not specified
            logger.warning(f"No hardware compatibility information found for {model_family} on {hardware_platform}")
            return {
                "compatibility_level": "limited",
                "description": "No specific compatibility information available."
            }
            
        finally:
            conn.close()
    
    def _extract_variables(self, template_content: str) -> Dict[str, Dict[str, Any]]:
        """
        Extract variables from template content.
        
        Args:
            template_content: Template content string
            
        Returns:
            Dictionary of variables with metadata
        """
        # Find all variables in the format ${variable_name}
        pattern = r'\${([a-zA-Z0-9_]+)}'
        matches = re.findall(pattern, template_content)
        
        # Create a dictionary of unique variables
        variables = {}
        for var_name in matches:
            if var_name not in variables:
                variables[var_name] = {
                    "description": f"Variable for {var_name}",
                    "required": True
                }
                
        return variables
        
    def render_template(self, 
                        template_id: str, 
                        variables: Dict[str, Any],
                        render_dependencies: bool = True) -> str:
        """
        Render a template with variables.
        
        Args:
            template_id: ID of the template to render
            variables: Dictionary of variable values
            render_dependencies: Whether to render dependencies
            
        Returns:
            Rendered template content
        """
        # Connect to the database
        conn = duckdb.connect(self.db_path)
        
        try:
            # Get the template
            result = conn.execute("""
                SELECT template_content, parent_template_id FROM templates WHERE template_id = ?
            """, [template_id]).fetchone()
            
            if not result:
                raise ValueError(f"Template not found with ID: {template_id}")
                
            template_content, parent_template_id = result
            
            # If this template has a parent, fetch and render the parent first
            if parent_template_id and render_dependencies:
                parent_content = self.render_template(parent_template_id, variables, render_dependencies)
                
                # Merge the parent content with this template's content
                # (This is a simple approach - in a real implementation, we'd have more sophisticated inheritance)
                template_content = self._merge_templates(parent_content, template_content)
            
            # Render dependencies if requested
            if render_dependencies:
                # Get dependencies
                dependencies = conn.execute("""
                    SELECT t.template_id, t.template_content, td.dependency_type
                    FROM template_dependencies td
                    JOIN templates t ON td.depends_on_template_id = t.template_id
                    WHERE td.template_id = ?
                """, [template_id]).fetchall()
                
                # Render each dependency and incorporate based on dependency type
                for dep_id, dep_content, dep_type in dependencies:
                    rendered_dep = self.render_template(dep_id, variables, True)
                    
                    # Merge based on dependency type
                    if dep_type == "include":
                        # Simple inclusion at template variable placeholder
                        placeholder = f"${{include_{dep_id}}}"
                        template_content = template_content.replace(placeholder, rendered_dep)
                    elif dep_type == "extend":
                        # Extend the template (base class implementation)
                        template_content = self._merge_templates(rendered_dep, template_content)
            
            # First pass: Replace basic variables
            for var_name, var_value in variables.items():
                placeholder = f"${{{var_name}}}"
                template_content = template_content.replace(placeholder, str(var_value))
            
            # Check for variable transformations but don't replace them yet
            # These will be handled by the TemplateRenderer using _process_variable_transforms
                
            # Check for missing plain variables (not transformations)
            missing_vars = re.findall(r'\${([a-zA-Z0-9_]+)}', template_content)
            if missing_vars:
                logger.warning(f"Template contains unreplaced variables: {missing_vars}")
                
            return template_content
            
        finally:
            conn.close()
    
    def _merge_templates(self, base_template: str, override_template: str) -> str:
        """
        Merge two templates for inheritance.
        
        Args:
            base_template: Base template content
            override_template: Template content that overrides the base
            
        Returns:
            Merged template content
        """
        # Simple implementation - just append
        # In a real implementation, we would have more sophisticated merging logic
        return base_template + "\n\n" + override_template
        
    def list_templates(self, 
                       model_family: Optional[str] = None,
                       template_type: Optional[str] = None,
                       hardware_platform: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List templates matching the given criteria.
        
        Args:
            model_family: Filter by model family (optional)
            template_type: Filter by template type (optional)
            hardware_platform: Filter by hardware platform (optional)
            
        Returns:
            List of matching templates
        """
        # Connect to the database
        conn = duckdb.connect(self.db_path)
        
        try:
            # Build query based on provided parameters
            query = "SELECT * FROM templates WHERE is_active = TRUE"
            params = []
            
            if model_family:
                query += " AND model_family = ?"
                params.append(model_family)
                
            if template_type:
                query += " AND template_type = ?"
                params.append(template_type)
                
            if hardware_platform:
                query += " AND (hardware_platform = ? OR hardware_platform IS NULL)"
                params.append(hardware_platform)
                
            # Order by model family, template type, and name
            query += " ORDER BY model_family, template_type, template_name"
            
            # Execute the query
            results = conn.execute(query, params).fetchall()
            
            # Convert to dictionaries
            templates = []
            for result in results:
                templates.append({
                    "template_id": result[0],
                    "template_name": result[1],
                    "template_type": result[2],
                    "model_family": result[3],
                    "hardware_platform": result[4],
                    "description": result[6],
                    "version": result[7],
                    "parent_template_id": result[8],
                    "created_at": result[9],
                    "updated_at": result[10]
                })
                
            logger.info(f"Listed {len(templates)} templates")
            return templates
            
        finally:
            conn.close()
            
    def get_compatible_hardware_platforms(self, model_family: str) -> List[Dict[str, Any]]:
        """
        Get compatible hardware platforms for a model family.
        
        Args:
            model_family: Model family
            
        Returns:
            List of compatible hardware platforms with compatibility level
        """
        # Connect to the database
        conn = duckdb.connect(self.db_path)
        
        try:
            # Query hardware compatibility
            results = conn.execute("""
                SELECT hardware_platform, compatibility_level, description
                FROM hardware_compatibility
                WHERE model_family = ? AND compatibility_level != 'none'
                ORDER BY 
                    CASE compatibility_level
                        WHEN 'full' THEN 1
                        WHEN 'limited' THEN 2
                        ELSE 3
                    END
            """, [model_family]).fetchall()
            
            # Convert to dictionaries
            platforms = []
            for result in results:
                platforms.append({
                    "hardware_platform": result[0],
                    "compatibility_level": result[1],
                    "description": result[2]
                })
                
            if not platforms:
                # If no specific compatibility information, return default platforms
                logger.warning(f"No hardware compatibility information found for {model_family}")
                return [
                    {"hardware_platform": "cpu", "compatibility_level": "full", "description": "Default compatibility"},
                    {"hardware_platform": "cuda", "compatibility_level": "limited", "description": "Default compatibility"}
                ]
                
            return platforms
            
        finally:
            conn.close()
            
    def generate_template_files(self,
                               model_name: str,
                               output_dir: str,
                               hardware_platform: Optional[str] = None,
                               template_types: Optional[List[str]] = None,
                               custom_variables: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
        """
        Generate template files for a model.
        
        Args:
            model_name: Name of the model
            output_dir: Directory to output the generated files
            hardware_platform: Hardware platform (optional)
            template_types: List of template types to generate (default: all types)
            custom_variables: Additional custom variables (optional)
            
        Returns:
            Dictionary of generated file paths by template type
        """
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Get model family
        model_family = self.get_model_family(model_name)
        if not model_family:
            raise ValueError(f"Could not determine model family for {model_name}")
            
        # Determine template types to generate
        if not template_types:
            template_types = ["skill", "test", "benchmark"]
            
        # Set up basic variables
        variables = {
            "model_name": model_name,
            "model_family": model_family,
            "hardware_type": hardware_platform or "cpu",
            "test_id": str(uuid.uuid4()),
            "batch_size": 1,
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        # Add custom variables
        if custom_variables:
            variables.update(custom_variables)
            
        # Generate each template type
        generated_files = {}
        
        for template_type in template_types:
            try:
                # Get template for this type
                template = self.get_template(
                    model_family=model_family,
                    template_type=template_type,
                    hardware_platform=hardware_platform
                )
                
                if not template:
                    logger.warning(f"No {template_type} template found for {model_family}")
                    continue
                    
                # Render the template
                rendered_content = self.render_template(
                    template_id=template["template_id"],
                    variables=variables,
                    render_dependencies=True
                )
                
                # Determine output file name
                if template_type == "skill":
                    filename = f"{model_name.replace('/', '_')}_{hardware_platform or 'cpu'}_skill.py"
                elif template_type == "test":
                    filename = f"test_{model_name.replace('/', '_')}_{hardware_platform or 'cpu'}.py"
                elif template_type == "benchmark":
                    filename = f"benchmark_{model_name.replace('/', '_')}_{hardware_platform or 'cpu'}.py"
                elif template_type == "documentation":
                    filename = f"{model_name.replace('/', '_')}_{hardware_platform or 'cpu'}_docs.md"
                else:
                    filename = f"{template_type}_{model_name.replace('/', '_')}_{hardware_platform or 'cpu'}.py"
                    
                # Write the rendered template to file
                output_path = os.path.join(output_dir, filename)
                with open(output_path, 'w') as f:
                    f.write(rendered_content)
                    
                generated_files[template_type] = output_path
                logger.info(f"Generated {template_type} template for {model_name}: {output_path}")
                
            except Exception as e:
                logger.error(f"Error generating {template_type} template for {model_name}: {e}")
                
        return generated_files

# Functions for initializing the database with default templates

def add_default_templates(db_path: str = DEFAULT_DB_PATH):
    """
    Add default templates to the database.
    
    Args:
        db_path: Path to the template database
    """
    # Create DB instance
    db = TemplateDatabase(db_path)
    
    # Add model family mappings
    add_default_model_mappings(db)
    
    # Add hardware compatibility
    add_default_hardware_compatibility(db)
    
    # Add text embedding templates
    add_text_embedding_templates(db)
    
    # Add vision templates
    add_vision_templates(db)
    
    # Add documentation templates
    add_documentation_templates(db)
    
def add_default_model_mappings(db: TemplateDatabase):
    """Add default model mappings to the database."""
    mappings = [
        ("bert-base-uncased", "text_embedding", "BERT base uncased model"),
        ("bert-large-uncased", "text_embedding", "BERT large uncased model"),
        ("roberta-base", "text_embedding", "RoBERTa base model"),
        ("gpt2", "text_generation", "GPT-2 small model"),
        ("t5-small", "text_generation", "T5 small model"),
        ("vit-base-patch16-224", "vision", "Vision Transformer base model"),
        ("whisper-tiny", "audio", "Whisper tiny model"),
        ("openai/clip-vit-base-patch32", "multimodal", "CLIP Vision-Text model")
    ]
    
    for model_name, model_family, description in mappings:
        db.add_model_mapping(model_name, model_family, description)
        
def add_default_hardware_compatibility(db: TemplateDatabase):
    """Add default hardware compatibility to the database."""
    compatibilities = [
        # Text embedding models
        ("text_embedding", "cpu", "full", "Full compatibility on CPU"),
        ("text_embedding", "cuda", "full", "Full compatibility on CUDA"),
        ("text_embedding", "rocm", "full", "Full compatibility on ROCm"),
        ("text_embedding", "mps", "full", "Full compatibility on MPS"),
        ("text_embedding", "openvino", "full", "Full compatibility on OpenVINO"),
        ("text_embedding", "qnn", "full", "Full compatibility on Qualcomm Neural Network"),
        ("text_embedding", "webnn", "full", "Full compatibility on WebNN"),
        ("text_embedding", "webgpu", "full", "Full compatibility on WebGPU"),
        
        # Text generation models
        ("text_generation", "cpu", "full", "Full compatibility on CPU"),
        ("text_generation", "cuda", "full", "Full compatibility on CUDA"),
        ("text_generation", "rocm", "limited", "Limited compatibility on ROCm"),
        ("text_generation", "mps", "limited", "Limited compatibility on MPS"),
        ("text_generation", "openvino", "limited", "Limited compatibility on OpenVINO"),
        ("text_generation", "qnn", "limited", "Limited compatibility on Qualcomm Neural Network"),
        ("text_generation", "webnn", "limited", "Limited compatibility on WebNN"),
        ("text_generation", "webgpu", "limited", "Limited compatibility on WebGPU"),
        
        # Vision models
        ("vision", "cpu", "full", "Full compatibility on CPU"),
        ("vision", "cuda", "full", "Full compatibility on CUDA"),
        ("vision", "rocm", "full", "Full compatibility on ROCm"),
        ("vision", "mps", "full", "Full compatibility on MPS"),
        ("vision", "openvino", "full", "Full compatibility on OpenVINO"),
        ("vision", "qnn", "full", "Full compatibility on Qualcomm Neural Network"),
        ("vision", "webnn", "full", "Full compatibility on WebNN"),
        ("vision", "webgpu", "full", "Full compatibility on WebGPU"),
        
        # Audio models
        ("audio", "cpu", "full", "Full compatibility on CPU"),
        ("audio", "cuda", "full", "Full compatibility on CUDA"),
        ("audio", "rocm", "limited", "Limited compatibility on ROCm"),
        ("audio", "mps", "limited", "Limited compatibility on MPS"),
        ("audio", "openvino", "limited", "Limited compatibility on OpenVINO"),
        ("audio", "qnn", "limited", "Limited compatibility on Qualcomm Neural Network"),
        ("audio", "webnn", "limited", "Limited compatibility on WebNN"),
        ("audio", "webgpu", "limited", "Limited compatibility on WebGPU"),
        
        # Multimodal models
        ("multimodal", "cpu", "full", "Full compatibility on CPU"),
        ("multimodal", "cuda", "full", "Full compatibility on CUDA"),
        ("multimodal", "rocm", "limited", "Limited compatibility on ROCm"),
        ("multimodal", "mps", "limited", "Limited compatibility on MPS"),
        ("multimodal", "openvino", "limited", "Limited compatibility on OpenVINO"),
        ("multimodal", "qnn", "limited", "Limited compatibility on Qualcomm Neural Network"),
        ("multimodal", "webnn", "limited", "Limited compatibility on WebNN"),
        ("multimodal", "webgpu", "limited", "Limited compatibility on WebGPU")
    ]
    
    for model_family, hardware, level, description in compatibilities:
        db.add_hardware_compatibility(model_family, hardware, level, description)

def add_text_embedding_templates(db: TemplateDatabase):
    """Add text embedding templates to the database."""
    # Load example template from distributed testing examples
    example_path = "/home/barberb/ipfs_accelerate_py/test/duckdb_api/distributed_testing/examples/text_embedding_template.py"
    
    try:
        with open(example_path, 'r') as f:
            template_content = f.read()
            
        # Add as skill template
        db.add_template(
            template_name="text_embedding_skill",
            template_type="skill",
            model_family="text_embedding",
            template_content=template_content,
            description="Basic template for text embedding model skills"
        )
        
        # Add test template based on the same structure
        test_template = template_content.replace("test_${model_family}", "test_${model_name.replace('-', '_').replace('/', '_')}")
        db.add_template(
            template_name="text_embedding_test",
            template_type="test",
            model_family="text_embedding",
            template_content=test_template,
            description="Basic template for text embedding model tests"
        )
        
        # Add benchmark template
        benchmark_template = template_content.replace(
            "def test_${model_name.replace('-', '_').replace('/', '_')}_on_${hardware_type}():",
            "def benchmark_${model_name.replace('-', '_').replace('/', '_')}_on_${hardware_type}(batch_size=${batch_size}):"
        )
        db.add_template(
            template_name="text_embedding_benchmark",
            template_type="benchmark",
            model_family="text_embedding",
            template_content=benchmark_template,
            description="Basic template for text embedding model benchmarks"
        )
    except Exception as e:
        logger.error(f"Error adding text embedding templates: {e}")

def add_vision_templates(db: TemplateDatabase):
    """Add vision templates to the database."""
    # Load example template from distributed testing examples
    example_path = "/home/barberb/ipfs_accelerate_py/test/duckdb_api/distributed_testing/examples/vision_template.py"
    
    try:
        with open(example_path, 'r') as f:
            template_content = f.read()
            
        # Add as skill template
        db.add_template(
            template_name="vision_skill",
            template_type="skill",
            model_family="vision",
            template_content=template_content,
            description="Basic template for vision model skills"
        )
        
        # Add test template based on the same structure
        test_template = template_content.replace("test_${model_family}", "test_${model_name.replace('-', '_').replace('/', '_')}")
        db.add_template(
            template_name="vision_test",
            template_type="test",
            model_family="vision",
            template_content=test_template,
            description="Basic template for vision model tests"
        )
        
        # Add benchmark template
        benchmark_template = template_content.replace(
            "def test_${model_name.replace('-', '_').replace('/', '_')}_on_${hardware_type}():",
            "def benchmark_${model_name.replace('-', '_').replace('/', '_')}_on_${hardware_type}(batch_size=${batch_size}):"
        )
        db.add_template(
            template_name="vision_benchmark",
            template_type="benchmark",
            model_family="vision",
            template_content=benchmark_template,
            description="Basic template for vision model benchmarks"
        )
    except Exception as e:
        logger.error(f"Error adding vision templates: {e}")

def add_documentation_templates(db: TemplateDatabase):
    """Add documentation templates to the database."""
    # Example documentation template
    doc_template = """# ${model_name} Implementation for ${hardware_type}

## Overview

This document describes the implementation of ${model_name} for ${hardware_type} hardware.

- **Model**: ${model_name}
- **Model Family**: ${model_family}
- **Hardware**: ${hardware_type}
- **Generation Date**: ${timestamp}

## Model Architecture

${model_name} is a ${model_family} model designed for ${model_family.replace('_', ' ')} tasks.

## Implementation Details

### Key Components

- Input processing
- Model initialization
- Inference optimization
- Output formatting

### Hardware-Specific Optimizations

The implementation includes optimizations specific to ${hardware_type} hardware:

${hardware_specific_optimizations}

## Usage Example

```python
# Import the skill
from ${model_name.replace('-', '_').replace('/', '_')}_${hardware_type}_skill import ${model_name.replace('-', '_').replace('/', '_')}Skill

# Create an instance
skill = ${model_name.replace('-', '_').replace('/', '_')}Skill()

# Set up the model
setup_success = skill.setup()

# Run inference
result = skill.run("This is a test input")
print(result)

# Clean up resources
skill.cleanup()
```

## Test Results

${test_results}

## Benchmark Results

${benchmark_results}

## Known Limitations

${limitations}
"""
    
    # Add documentation template
    db.add_template(
        template_name="general_documentation",
        template_type="documentation",
        model_family="text_embedding",  # Default family
        template_content=doc_template,
        description="Basic documentation template for model implementations"
    )
    
    # Add hardware-specific optimizations for different platforms
    hardware_optimizations = {
        "cpu": "- CPU threading optimizations\n- Cache-friendly operations\n- SSE/AVX instructions where applicable",
        "cuda": "- CUDA kernel optimizations\n- Mixed precision inference\n- Memory optimization for GPU",
        "webgpu": "- WebGPU shader optimizations\n- Browser-specific optimizations\n- Memory management for browser environment"
    }
    
    for hardware, optimizations in hardware_optimizations.items():
        # Create hardware-specific documentation templates
        hardware_doc_template = doc_template.replace("${hardware_specific_optimizations}", optimizations)
        
        db.add_template(
            template_name=f"documentation_{hardware}",
            template_type="documentation",
            model_family="text_embedding",  # Default family
            hardware_platform=hardware,
            template_content=hardware_doc_template,
            description=f"Documentation template with {hardware}-specific optimizations"
        )

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Template Database Management")
    parser.add_argument("--db-path", type=str, default=DEFAULT_DB_PATH, help="Path to the template database")
    parser.add_argument("--init", action="store_true", help="Initialize the database with default templates")
    parser.add_argument("--list", action="store_true", help="List templates in the database")
    parser.add_argument("--add-template", action="store_true", help="Add a template to the database")
    parser.add_argument("--template-name", type=str, help="Name of the template")
    parser.add_argument("--template-type", type=str, choices=TEMPLATE_TYPES, help="Type of template")
    parser.add_argument("--model-family", type=str, choices=MODEL_FAMILIES, help="Model family")
    parser.add_argument("--hardware", type=str, choices=HARDWARE_PLATFORMS, help="Hardware platform")
    parser.add_argument("--template-file", type=str, help="Path to template file")
    parser.add_argument("--description", type=str, help="Template description")
    parser.add_argument("--generate", action="store_true", help="Generate template files")
    parser.add_argument("--model", type=str, help="Model name for generation")
    parser.add_argument("--output-dir", type=str, default="./generated", help="Output directory for generated files")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Configure logging
    if args.verbose:
        logger.setLevel(logging.DEBUG)
        
    # Handle commands
    if args.init:
        logger.info(f"Initializing template database at {args.db_path}")
        add_default_templates(args.db_path)
        logger.info("Database initialized with default templates")
        
    elif args.list:
        db = TemplateDatabase(args.db_path, verbose=args.verbose)
        templates = db.list_templates(
            model_family=args.model_family,
            template_type=args.template_type,
            hardware_platform=args.hardware
        )
        
        print(f"Found {len(templates)} templates:")
        for i, template in enumerate(templates, 1):
            print(f"{i}. {template['template_name']} ({template['template_type']}, {template['model_family']})")
            if template['hardware_platform']:
                print(f"   Hardware: {template['hardware_platform']}")
            print(f"   Description: {template['description']}")
            print()
            
    elif args.add_template:
        if not args.template_name or not args.template_type or not args.model_family or not args.template_file:
            parser.error("--add-template requires --template-name, --template-type, --model-family, and --template-file")
            
        # Read template file
        with open(args.template_file, 'r') as f:
            template_content = f.read()
            
        # Add template
        db = TemplateDatabase(args.db_path, verbose=args.verbose)
        template_id = db.add_template(
            template_name=args.template_name,
            template_type=args.template_type,
            model_family=args.model_family,
            template_content=template_content,
            hardware_platform=args.hardware,
            description=args.description
        )
        
        print(f"Template added with ID: {template_id}")
        
    elif args.generate:
        if not args.model or not args.output_dir:
            parser.error("--generate requires --model and --output-dir")
            
        # Generate template files
        db = TemplateDatabase(args.db_path, verbose=args.verbose)
        generated_files = db.generate_template_files(
            model_name=args.model,
            output_dir=args.output_dir,
            hardware_platform=args.hardware
        )
        
        print(f"Generated {len(generated_files)} files:")
        for template_type, file_path in generated_files.items():
            print(f"- {template_type}: {file_path}")
    else:
        parser.print_help()